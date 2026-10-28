"""Camera calibration core (spec 03, T-105).

Pure functions over the PRD section 7 schemas: ground network from field
distances (4.1), RANSAC homography with FR-4 acceptance (4.3), pixel/meter
conversions (lens deferred, 4.4), crosswalk axis and movement (4.5, 4.6),
and zone validation as a pure function (4.5).

Only numpy and OpenCV are used, no SciPy (section 1). The runnable check
builds its fixture in code (section 6): no files, no GUI.
"""

from __future__ import annotations

import math
import re

import cv2
import numpy as np


def _mds_seed(ids, dist_of):
    """Classical-MDS layout over shortest-path distances, gauge-aligned.

    Fallback initializer when greedy circle seeding stalls: rigid networks
    whose links never offer two placed neighbors (e.g. bipartite K3,3
    sets). Raises ValueError on disconnected graphs or degenerate
    embeddings; LM and the rank test still decide the fit.
    """
    n = len(ids)
    idx = {pid: k for k, pid in enumerate(ids)}
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for (a, b), d in dist_of.items():
        if D[idx[a], idx[b]] > d:
            D[idx[a], idx[b]] = D[idx[b], idx[a]] = d
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    if not np.all(np.isfinite(D)):
        raise ValueError("disconnected distance graph")
    J = np.eye(n) - np.ones((n, n)) / n
    vals, vecs = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    top = np.argsort(vals)[::-1][:2]
    if vals[top[1]] <= 1e-12:
        raise ValueError("degenerate MDS embedding")
    X = vecs[:, top] * np.sqrt(np.maximum(vals[top], 0.0))
    X = X - X[idx[ids[0]]]
    ang = math.atan2(X[idx[ids[1]], 1], X[idx[ids[1]], 0])
    c, s = math.cos(ang), math.sin(ang)
    Xr = np.column_stack([c * X[:, 0] + s * X[:, 1], -s * X[:, 0] + c * X[:, 1]])
    return {pid: (float(Xr[idx[pid], 0]), float(Xr[idx[pid], 1])) for pid in ids}


def _circle_pair(c0, r0, c1, r1):
    """The two intersections of circles (c0, r0) and (c1, r1)."""
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    d = math.hypot(dx, dy)
    if d < 1e-12:
        raise ValueError("coincident circle centers")
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    xm, ym = c0[0] + a * dx / d, c0[1] + a * dy / d
    return [(xm + h * dy / d, ym - h * dx / d), (xm - h * dy / d, ym + h * dx / d)]


def solve_network(points, distances):
    """Fit-only distances -> ({id: [X, Y]}, report) (spec 03 section 4.1).

    Gauge: ``points[0]`` is (0, 0), ``points[1]`` is (d12, 0), so there are
    2N - 3 unknowns; P1-P2 must be a measured fit distance. Only distances
    with ``control: false`` enter the fit. Solver: seeded circle
    intersections, then Levenberg-Marquardt in numpy. Raises ValueError
    when a point has fewer than 2 links or rank(J) < 2N - 3.
    """
    ids = [p["id"] for p in points]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate point ids")
    fit = [d for d in distances if not d.get("control", False)]
    p1, p2 = ids[0], ids[1]
    d12 = None
    for d in fit:
        if {d["a"], d["b"]} == {p1, p2}:
            d12 = float(d["measured_m"])
    if d12 is None or d12 <= 0:
        raise ValueError("P1-P2 must be a measured fit distance (it defines +X)")
    dist_of = {}
    for d in fit:
        dist_of[(d["a"], d["b"])] = dist_of[(d["b"], d["a"])] = float(d["measured_m"])

    placed = {p1: (0.0, 0.0), p2: (d12, 0.0)}
    remaining = [i for i in ids if i not in placed]
    while remaining:
        progressed = False
        for pid in list(remaining):
            links = [(q, dist_of[(pid, q)]) for q in placed if (pid, q) in dist_of]
            if len(links) >= 2:
                cands = _circle_pair(placed[links[0][0]], links[0][1],
                                     placed[links[1][0]], links[1][1])
                if len(links) >= 3:
                    q3, d3 = links[2]
                    cands = sorted(cands, key=lambda c: abs(math.dist(c, placed[q3]) - d3))
                    placed[pid] = cands[0]
                else:
                    placed[pid] = max(cands, key=lambda c: c[1])
                remaining.remove(pid)
                progressed = True
                break
        if not progressed:
            # Greedy circle seeding stalls on rigid networks whose links
            # never offer two placed neighbors (e.g. bipartite K3,3 sets).
            # Fall back to a classical-MDS layout over shortest-path
            # distances, gauge-aligned; LM and the rank test still decide.
            try:
                placed = _mds_seed(ids, dist_of)
            except ValueError:
                raise ValueError(f"points with fewer than 2 links: {remaining}") from None
            remaining = []

    # Gauge: p1 fixed, p2 = (X2, 0) with X2 free; the P1-P2 fit distance
    # constrains it, so unknowns are 2N - 3 and only 3 columns drop out.
    order = [i for i in ids if i != p1]
    x = np.array([placed[p2][0]]
                 + [c for pid in order if pid != p2 for c in placed[pid]], dtype=float)

    def cols(pid):
        if pid == p1:
            return ()
        if pid == p2:
            return (0,)
        k = order.index(pid)
        return (2 * k - 1, 2 * k)

    def res_jac(x):
        P = {p1: (0.0, 0.0), p2: (x[0], 0.0)}
        for pid in order:
            if pid != p2:
                k = order.index(pid)
                P[pid] = (x[2 * k - 1], x[2 * k])
        r = np.zeros(len(fit))
        J = np.zeros((len(fit), 2 * len(ids) - 3))
        for k, d in enumerate(fit):
            pa = np.array(P[d["a"]])
            pb = np.array(P[d["b"]])
            diff = pa - pb
            L = float(np.linalg.norm(diff))
            u = diff / max(L, 1e-12)
            r[k] = L - float(d["measured_m"])
            for pid, s in ((d["a"], 1.0), (d["b"], -1.0)):
                for j, c in zip(cols(pid), u):
                    J[k, j] = s * c
        return r, J

    r, J = res_jac(x)
    prev = float(r @ r)
    lam = 1e-3
    for _ in range(100):
        JTJ = J.T @ J
        try:
            delta = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), -(J.T @ r))
        except np.linalg.LinAlgError:
            raise ValueError("singular normal equations") from None
        xn = x + delta
        rn, Jn = res_jac(xn)
        if float(rn @ rn) < prev:
            x, r, J, prev = xn, rn, Jn, float(rn @ rn)
            lam /= 3
        else:
            lam *= 3
        if float(np.linalg.norm(delta)) < 1e-6:
            break
    rank = int(np.linalg.matrix_rank(J))
    if rank < 2 * len(ids) - 3:
        raise ValueError(f"network not rigid: rank {rank} < {2 * len(ids) - 3}")
    coords = {p1: [0.0, 0.0], p2: [float(x[0]), 0.0]}
    for pid in order:
        if pid != p2:
            k = order.index(pid)
            coords[pid] = [float(x[2 * k - 1]), float(x[2 * k])]
    residuals = []
    for d in fit:
        f = math.dist(coords[d["a"]], coords[d["b"]])
        residuals.append({"a": d["a"], "b": d["b"], "measured_m": float(d["measured_m"]),
                          "fitted_m": f, "residual_m": f - float(d["measured_m"])})
    n, m = len(ids), len(fit)
    report = {"residuals": residuals,
              "rms_m": float(np.sqrt(np.mean([r["residual_m"] ** 2 for r in residuals]))),
              "max_m": max(abs(r["residual_m"]) for r in residuals),
              "redundancy": m - (2 * n - 3), "rank": rank,
              "warnings": [] if m - (2 * n - 3) > 0 else
              ["zero redundancy: every residual is zero by construction"]}
    return coords, report


def fit_homography(px, m, thresh_m=0.2):
    """RANSAC homography px -> m (spec 03 section 4.3).

    ``px`` and ``m`` are (N, 2); the threshold is in meters, the units of
    the destination points. Returns H (3x3, H[2][2] = 1) and a bool
    inlier mask. OpenCV refines H on the inliers; an empty H raises.
    """
    px = np.asarray(px, dtype=float).reshape(-1, 2)
    m = np.asarray(m, dtype=float).reshape(-1, 2)
    if len(px) < 4:
        raise ValueError(f"need >= 4 picked points, got {len(px)}")
    H, mask = cv2.findHomography(px, m, cv2.RANSAC, thresh_m)
    if H is None:
        raise ValueError("findHomography failed")
    return H / H[2, 2], mask.reshape(-1).astype(bool)


def _apply_H(H, pts):
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    q = (H @ np.vstack([p.T, np.ones(len(p))]))
    w = q[2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography maps a point to infinity")
    return (q[:2] / w).T


def to_ground(calib, pts_px):
    """(N, 2) px -> (N, 2) m with calib H (spec 03 section 5).

    Raises when ``lens`` is set: undistortion is deferred to the
    checkerboard fallback (section 4.4) and is not built yet.
    """
    if calib.get("lens") is not None:
        raise ValueError("lens undistortion is deferred (spec 03 section 4.4)")
    return _apply_H(np.asarray(calib["H"], dtype=float), pts_px)


def to_pixels(calib, pts_m):
    """(N, 2) m -> (N, 2) px with the inverse H; overlays only."""
    return _apply_H(np.linalg.inv(np.asarray(calib["H"], dtype=float)), pts_m)


def calib_ok(calib):
    """FR-4 from a calib.json dict alone (spec 03 section 4.3).

    Passes when reprojection RMSE over all picked points is <= 0.30 m,
    every control error_pct is <= 15, and >= 6 picked points are inliers
    (error <= ransac_thresh_m).
    """
    pts = [p for p in calib["points"] if p["px"] is not None]
    if not pts:
        return False
    err = np.linalg.norm(to_ground(calib, [p["px"] for p in pts])
                         - np.array([p["m"] for p in pts]), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    inliers = int(np.sum(err <= float(calib.get("ransac_thresh_m", 0.2))))
    controls = all(float(c["error_pct"]) <= 15 for c in calib.get("control", []))
    return rmse <= 0.30 and controls and inliers >= 6


def crosswalk_axis(polygon_m):
    """Curb-to-curb axis (spec 03 section 4.5): origin mid(v0, v1), unit
    vector toward mid(v2, v3), and length_m. Requires exactly 4 vertices."""
    v = np.asarray(polygon_m, dtype=float).reshape(-1, 2)
    if len(v) != 4:
        raise ValueError(f"crosswalk needs exactly 4 vertices, got {len(v)}")
    o = (v[0] + v[1]) / 2
    d = (v[2] + v[3]) / 2 - o
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        raise ValueError("degenerate crosswalk axis")
    return o, d / L, L


def _parse_gate(gid):
    m = re.fullmatch(r"G-([NESW])-(in|out)", gid)
    return (m.group(1), m.group(2)) if m else (None, None)


def gate_sequence(track_px, gates, min_frames=2):
    """Gate ids hit in time order, consecutive repeats collapsed (4.6).

    A hit needs the ground point inside the gate polygon for at least
    ``min_frames`` consecutive frames (default 2).
    """
    t = np.asarray(track_px, dtype=float).reshape(-1, 2)
    hits = []
    for g in gates:
        poly = np.asarray(g["polygon_px"], dtype=np.float32)
        inside = [cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                  for x, y in t]
        i = 0
        while i < len(inside):
            if inside[i]:
                j = i
                while j + 1 < len(inside) and inside[j + 1]:
                    j += 1
                if j - i + 1 >= min_frames:
                    hits.append((i, g["id"]))
                i = j + 1
            else:
                i += 1
    hits.sort()
    seq = []
    for _, gid in hits:
        if not seq or seq[-1] != gid:
            seq.append(gid)
    return seq


def movement(gate_ids):
    """Gate sequence -> movement label (spec 03 section 4.6).

    ``[G-W-in, G-S-out]`` -> ``EB-right``; missing in/out gate -> ``unknown``.
    """
    parsed = [_parse_gate(g) for g in gate_ids]
    first = next((p for p in parsed if p[1] == "in"), None)
    out = next((p for p in parsed[parsed.index(first):] if p[1] == "out"), None) \
        if first is not None else None
    if first is None or out is None:
        return "unknown"
    a, b = first[0], out[0]
    turn = ["uturn", "left", "thru", "right"][("NESW".index(b) - "NESW".index(a)) % 4]
    return {"N": "SB", "E": "WB", "S": "NB", "W": "EB"}[a] + "-" + turn


def _convex(poly):
    """True when every vertex lies on one closed side of every directed
    edge (spec 03 section 4.5). Rejects degenerate (zero-area, repeated
    vertices) and self-crossing polygons that consecutive-turn tests
    accept, such as collinear chains and consistently-turning stars."""
    pts = [(float(x), float(y)) for x, y in poly]
    n = len(pts)
    if n < 3 or len(set(pts)) != n:
        return False
    sign = 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        for j in range(n):
            if j == i or j == (i + 1) % n:
                continue
            cx, cy = pts[j]
            z = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(z) > 1e-9:
                s = 1 if z > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
    return sign != 0


def _exit_leg(entry_leg, turn):
    return "NESW"[("NESW".index(entry_leg)
                   + ["uturn", "left", "thru", "right"].index(turn)) % 4]


_ENTRY_OF = {"EB": "W", "WB": "E", "NB": "S", "SB": "N"}


def validate_zones(crosswalks, gates, conflict_areas, frame_size):
    """Hard zone checks (spec 03 section 4.5); returns warnings.

    Fails on ids outside their collection namespace, duplicated ids,
    approach/dir/crosswalk/movement fields that disagree with the id, a CA
    whose id does not derive from its movement, a CA on the wrong leg,
    polygons with < 3 vertices or non-convex (half-plane test, which also
    rejects degenerate and self-crossing shapes), crosswalks without
    exactly 4 vertices, and overlapping gates. Warns on CA vertices
    outside their crosswalk.
    """
    seen = set()

    def claim(i, pattern, kind):
        if not re.fullmatch(pattern, i):
            raise ValueError(f"bad {kind} id {i}")
        if i in seen:
            raise ValueError(f"duplicated zone id {i}")
        seen.add(i)

    for c in crosswalks:
        claim(c["id"], r"XW-[NESW]", "crosswalk")
        v = [tuple(p) for p in c["polygon_px"]]
        if len(v) != 4:
            raise ValueError(f"{c['id']}: crosswalk needs exactly 4 vertices")
        if not _convex(v):
            raise ValueError(f"{c['id']}: non-convex polygon")
    xw_leg = {c["id"]: c["id"].split("-")[1] for c in crosswalks}
    for g in gates:
        claim(g["id"], r"G-[NESW]-(in|out)", "gate")
        leg, direction = g["id"].split("-")[1:]
        if g["approach"] != leg or g["dir"] != direction:
            raise ValueError(f"{g['id']}: approach/dir disagree with id")
        v = [tuple(p) for p in g["polygon_px"]]
        if len(v) < 3:
            raise ValueError(f"{g['id']}: polygon needs >= 3 vertices")
        if not _convex(v):
            raise ValueError(f"{g['id']}: non-convex polygon")
    w, h = frame_size
    mask = np.zeros((h, w), dtype=np.int32)
    for k, g in enumerate(gates, start=1):
        layer = np.zeros((h, w), dtype=np.int32)
        cv2.fillPoly(layer, [np.asarray(g["polygon_px"], dtype=np.int32)], 1)
        if np.any(mask & layer):
            raise ValueError(f"{g['id']}: overlaps another gate")
        mask |= layer
    warnings = []
    for ca in conflict_areas:
        claim(ca["id"], r"CA-[NESW]-(EB|WB|NB|SB)[RLTU]", "conflict area")
        leg = ca["id"].split("-")[1]
        m = re.fullmatch(r"(EB|WB|NB|SB)-(right|left|thru|uturn)", ca["movement"])
        if not m or ca["crosswalk"] not in xw_leg:
            raise ValueError(f"{ca['id']}: bad movement or crosswalk")
        code = {"right": "R", "left": "L", "thru": "T", "uturn": "U"}[m.group(2)]
        if ca["id"] != f"CA-{leg}-{m.group(1)}{code}":
            raise ValueError(f"{ca['id']}: movement disagrees with id")
        if xw_leg[ca["crosswalk"]] != leg:
            raise ValueError(f"{ca['id']}: not on its crosswalk leg")
        entry, turn = _ENTRY_OF[m.group(1)], m.group(2)
        if leg != _exit_leg(entry, turn) and not (turn == "right" and leg == entry):
            raise ValueError(f"{ca['id']}: wrong leg for {ca['movement']}")
        v = [tuple(p) for p in ca["polygon_px"]]
        if len(v) < 3:
            raise ValueError(f"{ca['id']}: polygon needs >= 3 vertices")
        if not _convex(v):
            raise ValueError(f"{ca['id']}: non-convex polygon")
        xw_poly = np.asarray(next(c["polygon_px"] for c in crosswalks
                                   if c["id"] == ca["crosswalk"]), dtype=np.float32)
        if any(cv2.pointPolygonTest(xw_poly, (float(x), float(y)), False) < 0
               for x, y in v):
            warnings.append(f"{ca['id']}: vertex outside its crosswalk")
    return warnings


def _fixture():
    """Spec 03 section 6 fixture, built in code: 12 ground points in 4
    corner groups, 24 exact fit distances, controls C1 and C2, a physical
    1920x1080 pinhole camera (f = 1400 px), 20 m up and 12 m south of the
    30x30 m box, pitched 45 degrees down, and a 12x3 m crosswalk."""
    truth = {"P1": (0, 0), "P2": (4, 0), "P3": (0, 4),
             "P4": (26, 0), "P5": (30, 0), "P6": (26, 4),
             "P7": (26, 26), "P8": (30, 26), "P9": (30, 30),
             "P10": (0, 26), "P11": (4, 26), "P12": (0, 30)}
    ids = [f"P{i}" for i in range(1, 13)]
    sides = [("P1", "P2"), ("P1", "P3"), ("P2", "P3"),
             ("P4", "P5"), ("P4", "P6"), ("P5", "P6"),
             ("P7", "P8"), ("P7", "P9"), ("P8", "P9"),
             ("P10", "P11"), ("P10", "P12"), ("P11", "P12")]
    links = [("P3", "P4"), ("P3", "P5"), ("P2", "P4"),
             ("P4", "P8"), ("P6", "P7"), ("P6", "P8"),
             ("P9", "P10"), ("P7", "P12"), ("P8", "P10"),
             ("P10", "P1"), ("P12", "P3"), ("P11", "P1")]
    methods = ["tape", "laser", "stripes"]
    distances = [{"a": a, "b": b, "measured_m": math.dist(truth[a], truth[b]),
                  "method": methods[k % 3], "control": False}
                 for k, (a, b) in enumerate(sides + links)]
    controls = [("C1", "P2", "P7"), ("C2", "P5", "P11")]
    for cid, a, b in controls:
        distances.append({"a": a, "b": b, "measured_m": math.dist(truth[a], truth[b]),
                          "method": "laser", "control": True})
    sq = math.sqrt(2) / 2
    K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1.0]])
    R = np.array([[1, 0, 0], [0, -sq, -sq], [0, sq, -sq]])
    C = np.array([15.0, -12.0, 20.0])
    H_true = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
    picks = {}
    for pid in ids:
        q = H_true @ np.array([*truth[pid], 1.0])
        picks[pid] = [float(q[0] / q[2]), float(q[1] / q[2])]
    picks["P9"][1] += 60.0
    xw_m = [(9.0, 0.0), (9.0, 3.0), (21.0, 3.0), (21.0, 0.0)]
    q = (H_true @ np.vstack([np.array(xw_m).T, np.ones(4)]))
    xw_px = [(float(q[0, i] / q[2, i]), float(q[1, i] / q[2, i])) for i in range(4)]
    return ids, truth, distances, controls, H_true, picks, xw_m, xw_px


def _calib_dict(H, ids, truth, picks, controls, distances):
    meas = {(d["a"], d["b"]): d["measured_m"] for d in distances}
    control = []
    for cid, a, b in controls:
        pred = math.dist(to_ground({"H": H, "lens": None}, [picks[a]])[0],
                         to_ground({"H": H, "lens": None}, [picks[b]])[0])
        m = meas[(a, b)]
        control.append({"id": cid, "a": a, "b": b, "measured_m": m,
                        "predicted_m": pred, "error_pct": 100 * abs(pred - m) / m})
    pts = [p for p in
           [{"id": pid, "px": picks[pid], "m": list(truth[pid])} for pid in ids]]
    err = [math.dist(to_ground({"H": H, "lens": None}, [p["px"]])[0], p["m"])
           for p in pts if p["px"] is not None]
    rmse = math.sqrt(sum(e * e for e in err) / len(err))
    return {"H": np.asarray(H).tolist(), "ransac_thresh_m": 0.2, "method": "ransac",
            "lens": None, "points": pts, "control": control,
            "reprojection_rmse_m": rmse}


def check():
    """Spec 03 section 6: the 7 asserts; raises AssertionError."""
    ids, truth, distances, controls, H_true, picks, xw_m, xw_px = _fixture()
    order = {pid: k for k, pid in enumerate(ids)}

    # 1. Network: 0.01 m, P3 not mirrored, rank 2N - 3; one link cut raises.
    coords, report = solve_network([{"id": pid} for pid in ids], distances)
    assert max(math.dist(coords[pid], truth[pid]) for pid in ids) <= 0.01, report
    assert coords["P3"][1] > 0, coords["P3"]
    assert report["rank"] == 2 * 12 - 3, report
    # 1b. Fallback: bipartite K3,3 links stall greedy seeding (no point
    # ever has two placed neighbors) but still solve through MDS seeding.
    k_truth = {"P1": (0.0, 0.0), "P2": (6.0, 1.0), "P3": (2.0, 5.0),
               "P4": (5.0, 6.0), "P5": (-1.0, 8.0), "P6": (7.0, 4.0)}
    k_ids = ["P1", "P2", "P3", "P4", "P5", "P6"]
    k_dist = [{"a": a, "b": b, "measured_m": math.dist(k_truth[a], k_truth[b]),
               "method": "tape", "control": False}
              for a in ["P1", "P3", "P5"] for b in ["P2", "P4", "P6"]]
    _, k_rep = solve_network([{"id": p} for p in k_ids], k_dist)
    assert k_rep["rank"] == 2 * 6 - 3, k_rep
    assert k_rep["max_m"] <= 0.01, k_rep
    cut = [d for d in distances
           if not ({d["a"], d["b"]} in ({"P10", "P12"}, {"P7", "P12"}, {"P12", "P3"}))]
    try:
        solve_network([{"id": pid} for pid in ids], cut)
    except ValueError:
        pass
    else:
        raise AssertionError("P12 with one distance accepted")

    # 2. RANSAC flags exactly P9.
    px = np.array([picks[pid] for pid in ids])
    m = np.array([truth[pid] for pid in ids])
    H, mask = fit_homography(px, m)
    assert mask.sum() == 11 and not mask[order["P9"]], mask

    # 3. Homography: 1 m grid over the box, error <= 0.01 m everywhere.
    # The grid reaches px through the true camera (H_true goes m -> px);
    # to_pixels inverts, so it cannot take H_true (spec 03 section 5).
    grid = np.array([[float(x), float(y)] for y in range(31) for x in range(31)])
    q = H_true @ np.vstack([grid.T, np.ones(len(grid))])
    grid_px = (q[:2] / q[2]).T
    back = to_ground({"H": H, "lens": None}, grid_px)
    assert float(np.max(np.linalg.norm(back - grid, axis=1))) <= 0.01
    # to_pixels inverts the fitted H (overlays only, section 5).
    px_back = to_pixels({"H": H, "lens": None}, grid)
    assert float(np.max(np.linalg.norm(px_back - grid_px, axis=1))) <= 0.01

    # 4. Controls: |predicted - measured| <= 0.01 m, error_pct ~= 0.
    for cid, a, b in controls:
        pred = math.dist(to_ground({"H": H, "lens": None}, [picks[a]])[0],
                         to_ground({"H": H, "lens": None}, [picks[b]])[0])
        meas = math.dist(truth[a], truth[b])
        assert abs(pred - meas) <= 0.01, (cid, pred, meas)

    # 5. Acceptance: bad pick fails FR-4; px null passes.
    bad = _calib_dict(H, ids, truth, picks, controls, distances)
    assert bad["reprojection_rmse_m"] > 0.30, bad["reprojection_rmse_m"]
    assert not calib_ok(bad)
    good_picks = dict(picks)
    good_picks["P9"] = None
    Hg, _ = fit_homography(np.array([picks[pid] for pid in ids if pid != "P9"]),
                           np.array([truth[pid] for pid in ids if pid != "P9"]))
    good = _calib_dict(Hg, ids, truth, good_picks, controls, distances)
    assert good["reprojection_rmse_m"] <= 0.01, good["reprojection_rmse_m"]
    assert calib_ok(good)

    # 6. Zone: length_m = 12.00 +- 0.01, axis along the 12 m side.
    o, u, L = crosswalk_axis(np.array(xw_m))
    assert abs(L - 12.0) <= 0.01, L
    assert abs(float(u @ np.array([1.0, 0.0])) - 1.0) <= 0.01, u

    # 7. Movement and zone ids.
    assert movement(["G-W-in", "G-S-out"]) == "EB-right"
    assert movement(["G-N-in", "G-E-out"]) == "SB-left"
    assert movement(["G-S-in", "G-N-out"]) == "NB-thru"
    assert movement(["G-E-in"]) == "unknown"
    cx = sum(p[0] for p in xw_px) / 4
    cy = sum(p[1] for p in xw_px) / 4
    ca_px = [[cx + 0.2 * (x - cx), cy + 0.2 * (y - cy)] for x, y in xw_px]
    ok_warn = validate_zones(
        [{"id": "XW-S", "polygon_px": xw_px}],
        [{"id": "G-W-in", "approach": "W", "dir": "in",
          "polygon_px": [[10, 300], [100, 300], [100, 380], [10, 380]]},
         {"id": "G-S-out", "approach": "S", "dir": "out",
          "polygon_px": [[400, 900], [500, 900], [500, 980], [400, 980]]}],
        [{"id": "CA-S-EBR", "crosswalk": "XW-S", "movement": "EB-right",
          "polygon_px": ca_px}], (1920, 1080))
    assert ok_warn == [], ok_warn
    try:
        validate_zones([{"id": "XW-N", "polygon_px": xw_px}], [],
                       [{"id": "CA-N-EBR", "crosswalk": "XW-N",
                         "movement": "EB-right", "polygon_px": ca_px}], (1920, 1080))
    except ValueError:
        pass
    else:
        raise AssertionError("CA-N-EBR on XW-N accepted")

    # 7b. Strict geometry and namespaces: star, collinear, bow-tie and
    # dart polygons are non-convex; ids must match their collection; the
    # CA id must derive from its movement.
    assert not _convex([(0, 3), (2, -3), (-3, 1), (3, 1), (-2, -3)]), "star"
    assert not _convex([(0, 0), (1, 1), (2, 2)]), "collinear"
    assert not _convex([(0, 0), (2, 2), (0, 2), (2, 0)]), "bow-tie"
    assert not _convex([(0, 0), (4, 0), (1, 1), (4, 4), (0, 4)]), "dart"
    assert _convex([(0, 0), (4, 0), (4, 3), (0, 3)])
    square = [[10, 10], [20, 10], [20, 20], [10, 20]]
    for cw, ga, ca, why in [
        ([{"id": "G-W-in", "polygon_px": square}], [], [], "gate-named crosswalk"),
        ([{"id": "XW-S", "polygon_px": xw_px}], [],
         [{"id": "CA-S-EBR", "crosswalk": "XW-S", "movement": "SB-thru",
           "polygon_px": ca_px}], "movement/id mismatch"),
    ]:
        try:
            validate_zones(cw, ga, ca, (1920, 1080))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} accepted")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Camera calibration core (spec 03).")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        print("calib --check: PASS (network, RANSAC, H, controls, FR-4, zones, movement)")
        return
    parser.error("only --check is built in PR2; pick/draw/solve come in PR3")


if __name__ == "__main__":
    main()
