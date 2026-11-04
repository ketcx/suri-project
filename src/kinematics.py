"""Kinematics without metric calibration: PET in pixels (task 2).

Contract: spec 04 sections 3 (events.json), 4.4 (occupancy), 4.8
(pair_events) and 4.10 (clip window). Out of scope today: phases,
crossings, counterfactual, event_detail and flags. Speeds and movement
are null; X_m/Y_m stay empty.

Stdlib only, so `python3 src/kinematics.py` runs without the venv.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile

# PRD section 7, exactly.
TRACKS_COLUMNS = (
    "session_id",
    "chunk_id",
    "frame",
    "t_s",
    "obj_id",
    "cls",
    "x1",
    "y1",
    "x2",
    "y2",
    "gx_px",
    "gy_px",
    "X_m",
    "Y_m",
    "conf",
)

PERSON = "person"
CAR = "car"
CLASSES = (PERSON, CAR)

STATIC_PX = 40.0
STATIC_MIN_S = 10.0
MAX_GAP_FRAMES = 3
HYST_FRAMES = 2
MAX_PET_S = 3.0
SERIOUS_BELOW_S = 1.0

_SESSION_RE = re.compile(r"g(\d+)-")


def point_in_poly(pt, poly):
    """Ray casting; points on an edge count as inside."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if x1 == x2 and y1 == y2:
            continue
        if ((y1 <= y <= y2) or (y2 <= y <= y1)) and min(x1, x2) <= x <= max(x1, x2):
            if abs((y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)) < 1e-9:
                return True
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xin > x:
                inside = not inside
            elif xin == x:
                return True
    return inside


def _tenths(t_s):
    """Integer tenths of a second; all times compare exactly."""
    return int(round(float(t_s) * 10))


def load_tracks(path):
    """Read tracks.csv; group by obj_id, sort by t_s.

    Returns [{obj_id, cls, session_id,
    rows: [{frame, t_s, T, x, y, chunk_id}]}], with T integer tenths.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if list(reader.fieldnames or []) != list(TRACKS_COLUMNS):
            raise ValueError(f"tracks columns must be {list(TRACKS_COLUMNS)}")
        raw = list(reader)
    groups = {}
    for row in raw:
        if row["cls"] not in CLASSES:
            raise ValueError(f"bad cls {row['cls']!r} in {row['obj_id']!r}")
        groups.setdefault(row["obj_id"], []).append(row)
    tracks = []
    for obj_id, rows in groups.items():
        pts = sorted(
            ({"frame": int(r["frame"]), "t_s": float(r["t_s"]),
              "x": float(r["gx_px"]), "y": float(r["gy_px"]),
              "chunk_id": r["chunk_id"]} for r in rows),
            key=lambda p: (p["t_s"], p["frame"]),
        )
        for p in pts:
            p["T"] = _tenths(p["t_s"])
        tracks.append({"obj_id": obj_id, "cls": rows[0]["cls"],
                       "session_id": rows[0]["session_id"], "rows": pts})
    tracks.sort(key=lambda t: t["obj_id"])
    return tracks


def owned_ranges(chunks):
    """{chunk_id: [start_T, end_T)}; cut_k is the overlap midpoint (§4.1)."""
    ordered = sorted(chunks, key=lambda c: float(c["t0_s"]))
    owned = {}
    for c in ordered:
        owned[c["chunk_id"]] = [_tenths(c["t0_s"]),
                                _tenths(float(c["t0_s"]) + float(c["duration_s"]))]
    for a, b in zip(ordered, ordered[1:]):
        a_end = _tenths(float(a["t0_s"]) + float(a["duration_s"]))
        b_start = _tenths(b["t0_s"])
        if b_start < a_end:
            cut = (b_start + a_end) // 2
            owned[a["chunk_id"]][1] = cut
            owned[b["chunk_id"]][0] = cut
    return owned


def split_gaps(rows):
    """Split on gaps longer than 3 frames; fill shorter ones by lerp."""
    if not rows:
        return []
    segments, cur = [], [rows[0]]
    for prev, row in zip(rows, rows[1:]):
        gap = row["frame"] - prev["frame"] - 1
        if gap < 0:
            raise ValueError("rows out of order")
        if gap > MAX_GAP_FRAMES:
            segments.append(cur)
            cur = [row]
            continue
        for f in range(prev["frame"] + 1, row["frame"]):
            a = (f - prev["frame"]) / (row["frame"] - prev["frame"])
            cur.append({"frame": f,
                        "t_s": prev["t_s"] + a * (row["t_s"] - prev["t_s"]),
                        "x": prev["x"] + a * (row["x"] - prev["x"]),
                        "y": prev["y"] + a * (row["y"] - prev["y"]),
                        "chunk_id": row["chunk_id"]})
        cur.append(row)
    segments.append(cur)
    return segments


def is_static(segment):
    """True when the segment fits in a 40 px box and lasts >= 10 s."""
    xs = [p["x"] for p in segment]
    ys = [p["y"] for p in segment]
    return (max(xs) - min(xs) <= STATIC_PX
            and max(ys) - min(ys) <= STATIC_PX
            and segment[-1]["t_s"] - segment[0]["t_s"] >= STATIC_MIN_S)


FLAG_CODES = ("id_switch", "mask_jump", "impossible_speed", "occlusion",
              "paths_do_not_cross", "parked_or_static", "outside_phase_logic")
GAP_FRAMES = 3
T_EDGE_S = 0.5


def plausibility(event, tracks, zones=None, phases=None, rtor=None):
    """Spec 06 section 4.4. Pixel mode: only occlusion is computed.

    Returns {code: None | "first" | "second" | "both" | "pair"}. Metric
    flags stay None without calibration or phases, so they never reject.
    """
    flags = {code: None for code in FLAG_CODES}
    by_id = {t["obj_id"]: t for t in tracks}
    missing = []
    for role, t_key in (("first", "t_exit_s"), ("second", "t_enter_s")):
        track = by_id.get(event[role]["obj_id"])
        present = {p["T"] for p in track["rows"]} if track else set()
        edge = _tenths(event[role][t_key])
        half = _tenths(T_EDGE_S)
        gone = sum(1 for t in range(edge - half, edge + half + 1)
                   if t not in present)
        if gone >= GAP_FRAMES:
            missing.append(role)
    if missing:
        flags["occlusion"] = "both" if len(missing) == 2 else missing[0]
    return flags


def _confirmed_states(inside):
    """Hysteresis state per frame: flip after 2 in the new state.

    Returns (states, transitions). A transition is stamped at the first
    of the two consecutive frames, so the stamp frame itself already
    carries the new state; a lone flicker never flips.
    """
    confirmed = inside[0]
    pending = None
    transitions = []
    states = [confirmed]
    for i in range(1, len(inside)):
        if inside[i] == confirmed:
            pending = None
        elif pending is None:
            pending = (inside[i], i)
        elif i == pending[1] + 1:
            transitions.append((pending[1], pending[0]))
            states[pending[1]] = pending[0]
            confirmed = pending[0]
            pending = None
        states.append(confirmed)
    return states, transitions


def intervals_of(segment, obj_id, cls, area, poly, edge_first, edge_last):
    """Confirmed inside runs, clipped to inside frames, with edge types.

    enter is "cut" when the run starts at an owned-range edge, "track"
    when it starts inside after a gap split, "ok" when observed.
    Symmetric for exit. Touching a cut or the session edge censors.
    """
    n = len(segment)
    inside = [point_in_poly((p["x"], p["y"]), poly) for p in segment]
    states, _ = _confirmed_states(inside)
    out = []
    i = 0
    while i < n:
        if not states[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and states[j + 1]:
            j += 1
        while not inside[j]:
            j -= 1
        enter_idx, exit_idx = i, j
        enter_type = "cut" if edge_first else "track"
        if i > 0:
            enter_type = "ok"
        if exit_idx < n - 1:
            exit_type = "ok"
        else:
            exit_type = "cut" if edge_last else "track"
        out.append({
            "obj_id": obj_id, "cls": cls, "area": area,
            "t_enter": segment[enter_idx]["t_s"],
            "t_exit": segment[exit_idx]["t_s"],
            "f_enter": segment[enter_idx]["frame"],
            "f_exit": segment[exit_idx]["frame"],
            "enter_type": enter_type, "exit_type": exit_type,
        })
        i = j + 1
    return out


def pair_events(intervals, max_pet_s=MAX_PET_S):
    """Spec 04 section 4.8, exactly: PET per conflict area, both orders.

    Returns [(first, second, pet_s, simultaneous)]. A cut second entry
    never pairs. Non-simultaneous pairs need first.t_exit <=
    second.t_enter with pet < max_pet_s. Overlaps yield pet 0.0.
    """
    pairs = []
    areas = {}
    for iv in intervals:
        areas.setdefault(iv["area"], []).append(iv)
    for area, group in areas.items():
        for s in group:
            if s["enter_type"] == "cut":
                continue
            cands = [f for f in group
                     if f is not s and f["cls"] != s["cls"]
                     and f["t_exit"] <= s["t_enter"]]
            if not cands:
                continue
            f = max(cands, key=lambda c: (c["t_exit"], c["obj_id"]))
            pet = round(s["t_enter"] - f["t_exit"], 2)
            if f["exit_type"] != "cut" and pet < max_pet_s:
                pairs.append((f, s, pet, False))
        cars = [iv for iv in group if iv["cls"] == CAR]
        persons = [iv for iv in group if iv["cls"] == PERSON]
        for c in cars:
            overlapping = [p for p in persons
                           if p["t_enter"] <= c["t_exit"]
                           and c["t_enter"] <= p["t_exit"]]
            if not overlapping:
                continue
            p = min(overlapping,
                    key=lambda q: (-(min(q["t_exit"], c["t_exit"])
                                     - max(q["t_enter"], c["t_enter"])),
                                   q["obj_id"]))
            later = p if p["t_enter"] >= c["t_enter"] else c
            earlier = c if later is p else p
            if later["enter_type"] != "cut":
                pairs.append((earlier, later, 0.0, True))
    kept = {}
    for f, s, pet, sim in pairs:
        if not sim:
            key = id(f)
            if key not in kept or pet < kept[key][2]:
                kept[key] = (f, s, pet, sim)
    by_pair = {}
    for f, s, pet, sim in kept.values():
        person = f if f["cls"] == PERSON else s
        car = s if f["cls"] == PERSON else f
        key = (person["obj_id"], car["obj_id"])
        if key not in by_pair or pet < by_pair[key][2]:
            by_pair[key] = (f, s, pet, sim)
    sims = [p for p in pairs if p[3]]
    for f, s, pet, sim in sims:
        person = f if f["cls"] == PERSON else s
        car = s if f["cls"] == PERSON else f
        key = (person["obj_id"], car["obj_id"])
        if key not in by_pair or pet < by_pair[key][2]:
            by_pair[key] = (f, s, pet, sim)
    out = sorted(by_pair.values(), key=lambda p: (p[1]["t_enter"], p[1]["obj_id"]))
    return out


def clip_window(first, second, chunk_start, chunk_end):
    """Section 4.10 window, shifted (never shortened) into the chunk."""
    a = min(first["t_exit"], second["t_enter"]) - 2.0
    b = max(first["t_exit"], second["t_enter"]) + 2.0
    if b - a > 8.0:
        mid = second["t_enter"]
        a, b = mid - 4.0, mid + 4.0
    shift = 0.0
    if a < chunk_start:
        shift = chunk_start - a
    if b + shift > chunk_end:
        shift -= (b + shift) - chunk_end
    a, b = a + shift, b + shift
    if a < chunk_start or b > chunk_end:
        a, b = max(a, chunk_start), min(b, chunk_end)
    return [round(a, 2), round(b, 2)]


def load_zones(path):
    """Read zones.json; needs conflict_areas with polygon_px."""
    import json

    with open(path, encoding="utf-8") as fh:
        zones = json.load(fh)
    areas = zones.get("conflict_areas")
    if not areas:
        raise ValueError("zones.json needs conflict_areas")
    for area in areas:
        poly = area.get("polygon_px")
        if not poly or len(poly) < 3:
            raise ValueError(f"{area.get('id')}: polygon needs >= 3 vertices")
    return zones


def solve(tracks, chunks, zones):
    """Occupancy, pairing and events for one session. Returns a summary.

    Rows outside their chunk's owned range (§4.1) are dropped; a segment
    touching an owned edge gets a cut boundary there. Each event belongs
    to the chunk owning second.t_enter.
    """
    owned = owned_ranges(chunks)
    areas = {a["id"]: [tuple(p) for p in a["polygon_px"]]
             for a in zones["conflict_areas"]}
    static_dropped = 0
    outside_dropped = 0
    intervals = []
    kept_ts = {}
    for track in tracks:
        cid = track["rows"][0]["chunk_id"]
        if cid not in owned:
            raise ValueError(f"no owned range for chunk {cid!r}")
        start, end = owned[cid]
        kept = [r for r in track["rows"] if start <= r["T"] < end]
        outside_dropped += len(track["rows"]) - len(kept)
        if not kept:
            continue
        first_T = track["rows"][0]["T"]
        last_T = track["rows"][-1]["T"]
        segments = split_gaps(kept)
        for si, segment in enumerate(segments):
            if is_static(segment):
                static_dropped += 1
                continue
            edge_first = (si == 0 and (first_T < start
                                       or segment[0]["T"] == start))
            edge_last = (si == len(segments) - 1 and (last_T >= end
                         or segment[-1]["T"] == end - 1))
            for s in segment:
                kept_ts.setdefault(s["chunk_id"], []).append(s["t_s"])
            for area_id, poly in areas.items():
                intervals.extend(intervals_of(
                    segment, track["obj_id"], track["cls"], area_id, poly,
                    edge_first, edge_last))
    by_area = {}
    for iv in intervals:
        by_area.setdefault(iv["area"], []).append(iv)
    pairs = pair_events(intervals)
    sessions = sorted({t["session_id"] for t in tracks})
    events = []
    for session_id in sessions:
        match = _SESSION_RE.match(session_id)
        if not match:
            raise ValueError(f"bad session_id {session_id!r}")
        base = int(match.group(1)) * 1000 + 1
        mine = [p for p in pairs
                if _session_of(p, tracks) == session_id]
        if len(mine) > 999:
            raise ValueError(f"more than 999 events in {session_id}")
        for i, (first, second, pet, _sim) in enumerate(mine):
            chunk_id = _owner_of(second["t_enter"], owned)
            c_ts = kept_ts[chunk_id]
            window = clip_window(first, second, min(c_ts), max(c_ts))
            events.append({
                "event_id": f"E-{base + i:03d}",
                "session_id": session_id,
                "chunk_id": chunk_id,
                "conflict_area": first["area"],
                "first": {"obj_id": first["obj_id"], "cls": first["cls"],
                          "t_exit_s": first["t_exit"], "speed_mps": None,
                          "movement": None},
                "second": {"obj_id": second["obj_id"], "cls": second["cls"],
                           "t_enter_s": second["t_enter"], "speed_mps": None,
                           "movement": None},
                "pet_s": pet,
                "tier": "serious" if pet < SERIOUS_BELOW_S else "close",
                "cycle_idx": 0,
                "phase_at_conflict": None,
                "clip": f"clips/E-{base + i:03d}.mp4",
                "window_s": window,
                "status": "candidate",
                "verify": {"reason_codes": [], "tool_calls": []},
            })
    return {"static_dropped": static_dropped,
            "outside_dropped": outside_dropped, "intervals": intervals,
            "events": events,
            "by_area": {a: len(v) for a, v in by_area.items()}}


def _session_of(pair, tracks):
    for track in tracks:
        if track["obj_id"] in (pair[0]["obj_id"], pair[1]["obj_id"]):
            return track["session_id"]
    raise ValueError("interval without track")


def _owner_of(t_enter, owned):
    """Chunk id whose owned range holds second.t_enter (§4.1)."""
    want = _tenths(t_enter)
    for chunk_id, (start, end) in owned.items():
        if start <= want < end:
            return chunk_id
    raise ValueError(f"t_enter {t_enter} outside every owned range")


def _mk_track(obj_id, cls, session, chunk, f0, f1, xfn, yfn, t0=0.0):
    rows = [{"frame": f, "t_s": t0 + f / 10.0, "x": xfn(f), "y": yfn(f),
             "chunk_id": chunk} for f in range(f0, f1 + 1)]
    for r in rows:
        r["T"] = int(round(r["t_s"] * 10))
    return {"obj_id": obj_id, "cls": cls, "session_id": session, "rows": rows}


def _chunk(cid, t0, duration):
    return {"chunk_id": cid, "t0_s": t0, "duration_s": duration}


def _mk_zones():
    return {"conflict_areas": [
        {"id": "CA-E-CAR", "crosswalk": "XW-E", "movement": "EB-thru",
         "polygon_px": [[100, 0], [200, 0], [200, 60], [100, 60]]}]}


def check():
    """Synthetic scenes in pixels: PET 1.5 both orders, no-conflict + parked."""
    zones = _mk_zones()
    session, chunk = "g1-20261104", "c00"
    one = [_chunk(chunk, 0.0, 12.0)]
    # 1. Person first, true PET 1.5.
    person = _mk_track("p1", PERSON, session, chunk, 0, 100,
                       lambda f: 2.5 * f, lambda f: 30.0)
    car = _mk_track("c1", CAR, session, chunk, 90, 110,
                    lambda f: 150.0, lambda f: 1010.0 - 10 * f)
    summary = solve([person, car], one, zones)
    assert summary["static_dropped"] == 0
    assert len(summary["events"]) == 1
    event = summary["events"][0]
    assert abs(event["pet_s"] - 1.5) <= 0.1, event["pet_s"]
    assert event["tier"] == "close" and event["event_id"] == "E-1001"
    assert event["first"]["cls"] == "person" and event["status"] == "candidate"
    assert event["verify"] == {"reason_codes": [], "tool_calls": []}
    assert event["window_s"] == [5.5, 11.0], event["window_s"]
    # 2. Car first, the mirror case.
    car2 = _mk_track("c2", CAR, session, chunk, 0, 30,
                     lambda f: 150.0, lambda f: 200.0 - 10 * f)
    person2 = _mk_track("p2", PERSON, session, chunk, 0, 60,
                        lambda f: 2.5 * f + 12.5, lambda f: 30.0)
    summary = solve([person2, car2], one, zones)
    assert len(summary["events"]) == 1
    event = summary["events"][0]
    assert abs(event["pet_s"] - 1.5) <= 0.1, event["pet_s"]
    assert event["tier"] == "close"
    assert event["first"]["cls"] == "car"
    # 3. No conflict plus a parked car: 0 events, parked in the static log.
    car3 = _mk_track("c3", CAR, session, chunk, 115, 135,
                     lambda f: 150.0, lambda f: 1260.0 - 10 * f)
    parked = _mk_track("c4", CAR, session, chunk, 0, 120,
                       lambda f: 150.0, lambda f: 30.0)
    summary = solve([person, car3, parked], [_chunk(chunk, 0.0, 14.0)], zones)
    assert len(summary["events"]) == 0, summary["events"]
    assert summary["static_dropped"] == 1
    # 4. Overlap: the encounter straddles the cut, exactly one event.
    k0, k1 = "k0", "k1"
    two = [_chunk(k0, 0.0, 30.0), _chunk(k1, 29.0, 30.0)]
    p0 = _mk_track("p0", PERSON, session, k0, 270, 299,
                   lambda f: 5.0 * f - 1300.0, lambda f: 30.0)
    p1 = _mk_track("p1", PERSON, session, k1, 5, 20,
                   lambda f: 150.0 + 5 * f, lambda f: 30.0, t0=29.0)
    c1 = _mk_track("c1", CAR, session, k1, 15, 30,
                   lambda f: 150.0, lambda f: 260.0 - 10 * f, t0=29.0)
    summary = solve([p0, p1, c1], two, zones)
    assert len(summary["events"]) == 1, summary["events"]
    assert summary["outside_dropped"] == 5, summary["outside_dropped"]
    event = summary["events"][0]
    assert event["event_id"] == "E-1001"
    assert abs(event["pet_s"] - 1.0) <= 0.1, event["pet_s"]
    assert event["tier"] == "close"
    assert event["chunk_id"] == k1, event["chunk_id"]
    assert event["window_s"] == [29.5, 32.0], event["window_s"]
    by_obj = {(iv["obj_id"], iv["area"]): iv for iv in summary["intervals"]}
    assert by_obj[("p0", "CA-E-CAR")]["exit_type"] == "cut"
    assert by_obj[("p1", "CA-E-CAR")]["enter_type"] == "cut"
    # Edges: empty, cut second, PET boundaries, hysteresis, static edge.
    assert solve([], [], zones)["events"] == []
    cut_car = {"obj_id": "c", "cls": CAR, "area": "CA-E-CAR",
               "t_enter": 9.5, "t_exit": 10.0,
               "enter_type": "cut", "exit_type": "ok"}
    ok_person = {"obj_id": "p", "cls": PERSON, "area": "CA-E-CAR",
                 "t_enter": 4.0, "t_exit": 8.0,
                 "enter_type": "ok", "exit_type": "ok"}
    assert pair_events([ok_person, cut_car]) == []
    far = dict(cut_car, enter_type="ok", t_enter=11.0, t_exit=11.5)
    assert pair_events([ok_person, far]) == []
    near = dict(far, t_enter=9.0, t_exit=9.5)
    got = pair_events([ok_person, near])
    assert len(got) == 1 and got[0][2] == 1.0
    rounded_out = dict(far, t_enter=ok_person["t_exit"] + 2.996,
                       t_exit=ok_person["t_exit"] + 3.5)
    assert pair_events([ok_person, rounded_out]) == []
    serious_gap = dict(far, t_enter=ok_person["t_exit"] + 0.9,
                       t_exit=ok_person["t_exit"] + 1.4)
    got_s = pair_events([ok_person, serious_gap])
    assert len(got_s) == 1 and got_s[0][2] == 0.9, got_s
    assert ("serious" if got_s[0][2] < SERIOUS_BELOW_S else "close") == "serious"
    close_seg = [{"frame": f, "t_s": f / 10.0, "x": 0.0, "y": -10.0,
                  "chunk_id": chunk} for f in (0, 2, 3, 5)]
    assert [p["frame"] for p in split_gaps(close_seg)[0]] == [0, 1, 2, 3, 4, 5]
    far_seg = [{"frame": f, "t_s": f / 10.0, "x": 0.0, "y": -10.0,
                "chunk_id": chunk} for f in (0, 5)]
    assert len(split_gaps(far_seg)) == 2
    flicker = [{"frame": f, "t_s": f / 10.0,
                "x": 150.0 if f == 1 else 0.0, "y": 30.0, "chunk_id": chunk}
               for f in range(3)]
    poly = [(100, 0), (200, 0), (200, 60), (100, 60)]
    assert intervals_of(flicker, "p", PERSON, "CA-E-CAR", poly, False, False) == []
    steady = [{"frame": f, "t_s": f / 10.0, "x": 150.0, "y": 30.0,
               "chunk_id": chunk} for f in range(101)]
    assert is_static(steady) and not is_static(steady[:100])
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "tracks.csv")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(TRACKS_COLUMNS)
            writer.writerow(["g1-20261104", "c00", 1, 0.1,
                             "c00:person:1", "person",
                             0, 0, 10, 10, 5, 10, "", "", ""])
            writer.writerow(["g1-20261104", "c00", 0, 0.0,
                             "c00:person:1", "person",
                             0, 0, 10, 10, 5, 10, "", "", ""])
        loaded = load_tracks(csv_path)
        assert [r["frame"] for r in loaded[0]["rows"]] == [0, 1]
        assert [r["T"] for r in loaded[0]["rows"]] == [0, 1]
        with open(csv_path, "w", newline="") as fh:
            fh.write("a,b\n1,2\n")
        try:
            load_tracks(csv_path)
        except ValueError:
            pass
        else:
            raise AssertionError("bad header accepted")
    # Plausibility (spec 06 section 4.4): continuous tracks stay clean,
    # a gap at the PET endpoint names its actor, metric flags stay None.
    full_p = _mk_track("fp", PERSON, session, chunk, 40, 90,
                       lambda f: 150.0, lambda f: 30.0)
    full_c = _mk_track("fc", CAR, session, chunk, 75, 95,
                       lambda f: 150.0, lambda f: 50.0)
    evt = {"first": {"obj_id": "fp", "t_exit_s": 8.0},
           "second": {"obj_id": "fc", "t_enter_s": 8.5}}
    flags = plausibility(evt, [full_p, full_c])
    assert flags["occlusion"] is None, flags
    assert all(v is None for k, v in flags.items()), flags
    gap_rows = (_mk_track("gc", CAR, session, chunk, 85, 97,
                          lambda f: 150.0, lambda f: 50.0)["rows"]
                + _mk_track("gc", CAR, session, chunk, 101, 115,
                            lambda f: 150.0, lambda f: 50.0)["rows"])
    gap_c = {"obj_id": "gc", "cls": CAR, "session_id": session,
             "rows": gap_rows}
    evt = {"first": {"obj_id": "fp", "t_exit_s": 8.0},
           "second": {"obj_id": "gc", "t_enter_s": 9.5}}
    flags = plausibility(evt, [full_p, gap_c])
    assert flags == {c: (None if c != "occlusion" else "second")
                     for c in FLAG_CODES}, flags


def main(argv=None):
    parser = argparse.ArgumentParser(description="PET in pixels (task 2).")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        print("kinematics --check: PASS (pet both orders, static, edges)")
        return
    if not args.run_dir:
        parser.error("need RUN_DIR (or --check)")
    try:
        tracks = load_tracks(os.path.join(args.run_dir, "tracks.csv"))
        chunks_path = os.path.join(args.run_dir, "chunks.csv")
        with open(chunks_path, newline="") as fh:
            chunks = [{"chunk_id": r["chunk_id"], "t0_s": r["t0_s"],
                       "duration_s": r["duration_s"]}
                      for r in csv.DictReader(fh)]
        zones_path = os.path.join(args.run_dir, "zones.json")
        if not os.path.exists(zones_path):
            parser.error(f"zones.json missing: {zones_path}")
        zones = load_zones(zones_path)
        summary = solve(tracks, chunks, zones)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    import json

    with open(os.path.join(args.run_dir, "events.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary["events"], fh, indent=2, sort_keys=True)
        fh.write("\n")
    tiers = {"serious": 0, "close": 0}
    for event in summary["events"]:
        tiers[event["tier"]] += 1
    print(f"static_dropped={summary['static_dropped']}")
    for area, count in sorted(summary["by_area"].items()):
        print(f"intervals {area}={count}")
    print(f"events serious={tiers['serious']} close={tiers['close']}")
    for event in summary["events"]:
        print(f"{event['event_id']} {event['first']['cls']}->{event['second']['cls']} "
              f"pet={event['pet_s']} {event['tier']} {event['conflict_area']}")


if __name__ == "__main__":
    main()
