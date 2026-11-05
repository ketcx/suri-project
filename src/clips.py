"""Evidence clips: blurred 720p renders, keyframes and audit (spec 09).

Reads one session's events.json, tracks.csv, chunks.csv and zones.json,
cuts each event window from the 10 fps analysis chunk, blurs every
person box (mask fallback: whole box dilated by r, spec 09 section 8),
draws the legend overlays after the blur, and encodes 720p with no
audio and a blurred-v1 tag. audit() enforces the publishable shape.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

FPS = 10
HEIGHT = 720
CRF = 23
R_PX = 4
SIGMA = 8.0
MAX_IDS = 16

# BGR legend (spec 09 section 4.4).
CA_COLOR = (0, 255, 255)
FIRST_COLOR = (255, 255, 0)
SECOND_COLOR = (255, 0, 255)
OTHER_COLOR = (128, 128, 128)

FORMAT_TAGS = {"major_brand", "minor_version", "compatible_brands",
               "encoder", "comment"}
STREAM_TAGS = {"language", "handler_name", "vendor_id", "encoder"}


def clip_window(t_exit_s, t_enter_s):
    """Section 4.10 window: [min - 2, max + 2], capped at 8 s."""
    a = min(t_exit_s, t_enter_s) - 2.0
    b = max(t_exit_s, t_enter_s) + 2.0
    if b - a > 8.0:
        mid = t_enter_s
        a, b = mid - 4.0, mid + 4.0
    return [round(a, 2), round(b, 2)]


def dilate(mask, r=R_PX):
    """Dilate a bool mask by r px with an elliptical kernel."""
    import cv2
    import numpy as np

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def head_box(mask, frac=0.25):
    """Padded head box: top frac rows of the mask, half-width padded."""
    import numpy as np

    rows = np.flatnonzero(mask.any(1))
    if len(rows) == 0:
        return np.zeros_like(mask, dtype=bool)
    band_end = rows[0] + max(1, int(round((rows[-1] - rows[0] + 1) * frac)))
    cols = np.flatnonzero(mask[rows[0]:band_end].any(0))
    if len(cols) == 0:
        return np.zeros_like(mask, dtype=bool)
    pad = (cols[-1] - cols[0] + 1) / 2.0
    out = np.zeros_like(mask, dtype=bool)
    r0 = max(0, int(rows[0] - pad))
    r1 = min(mask.shape[0], band_end)
    c0 = max(0, int(cols[0] - pad))
    c1 = min(mask.shape[1], int(cols[-1] + pad) + 1)
    out[r0:r1, c0:c1] = True
    return out


def blur_frame(frame, person_masks, plate_masks, r=R_PX, sigma=SIGMA):
    """Blur person/plate regions; everything else byte-identical.

    With MAX_IDS person masks the whole frame is blurred (crowd rule:
    missed people cannot be ruled out).
    """
    import cv2
    import numpy as np

    if len(person_masks) >= MAX_IDS:
        region = np.ones(frame.shape[:2], dtype=bool)
    else:
        region = np.zeros(frame.shape[:2], dtype=bool)
        for mask in person_masks:
            region |= dilate(mask, r) | head_box(mask)
        for mask in plate_masks:
            region |= dilate(mask, r)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    return np.where(region[..., None], blurred, frame)


def box_mask(shape, x1, y1, x2, y2):
    """Full-frame bool mask of one box (the no-masks fallback, section 8)."""
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    mask[max(0, int(y1)):min(h, int(y2) + 1),
         max(0, int(x1)):min(w, int(x2) + 1)] = True
    return mask


def rect(mask_poly, shape):
    """Rasterize a polygon to a full-frame bool mask."""
    import cv2
    import numpy as np

    img = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(img, [np.asarray(mask_poly, dtype=np.int32)], 1)
    return img.astype(bool)


def overlay(base, poly, color, alpha=0.3):
    """Alpha-fill a polygon over base (BGR)."""
    import cv2
    import numpy as np

    layer = base.copy()
    cv2.fillPoly(layer, [np.asarray(poly, dtype=np.int32)], color)
    return cv2.addWeighted(layer, alpha, base, 1 - alpha, 0)


def draw_legend(frame, event, ca_poly, first_now, second_now, others,
                first_pts, second_pts):
    """Overlays after the blur: CA fill, actor boxes and 2 s trails, id."""
    import cv2

    frame = overlay(frame, ca_poly, CA_COLOR)
    for pts, color in ((first_pts, FIRST_COLOR), (second_pts, SECOND_COLOR)):
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2)
    for row in first_now:
        box(frame, row, FIRST_COLOR)
    for row in second_now:
        box(frame, row, SECOND_COLOR)
    for row in others:
        box(frame, row, OTHER_COLOR)
    cv2.putText(frame, event["event_id"], (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def trail(rows):
    """Last 2 s of ground points as int pixels for one actor's rows."""
    if not rows:
        return []
    cutoff = float(rows[-1]["t_s"]) - 2.0
    return [(int(float(r["gx_px"])), int(float(r["gy_px"])))
            for r in rows if float(r["t_s"]) >= cutoff]


def box(frame, row, color):
    """One thin box outline."""
    import cv2

    cv2.rectangle(frame, (int(float(row["x1"])), int(float(row["y1"]))),
                  (int(float(row["x2"])), int(float(row["y2"]))), color, 2)


def encode(frames, fps, out_path):
    """H264 pipe: downscale to 720p max, no audio, blurred-v1 tag."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{frames[0].shape[1]}x{frames[0].shape[0]}",
           "-r", str(fps), "-i", "-",
           "-vf", "scale=-2:'min(720,ih)'",
           "-c:v", "libx264", "-crf", str(23), "-pix_fmt", "yuv420p",
           "-an", "-map_metadata", "-1", "-map_metadata:s", "-1",
           "-metadata", "comment=blurred-v1", "-movflags", "+faststart",
           str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(b"".join(f.tobytes() for f in frames))
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"encode failed: {err.decode().strip()}")
    return out_path


def probe(path):
    """ffprobe format + streams as dicts."""
    cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams",
           "-of", "json", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def audit_file(path):
    """Problems in one published mp4; empty means pass."""
    problems = []
    try:
        info = probe(path)
    except RuntimeError as exc:
        return [str(exc)]
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            problems.append(f"{path}: audio stream present")
        if "height" in stream and int(stream["height"]) > HEIGHT:
            problems.append(f"{path}: height {stream['height']} > 720")
        for key in (stream.get("tags") or {}):
            if key not in STREAM_TAGS:
                problems.append(f"{path}: stream tag {key!r} not allowed")
    tags = (info.get("format") or {}).get("tags") or {}
    for key in tags:
        if key not in FORMAT_TAGS:
            problems.append(f"{path}: format tag {key!r} not allowed")
    if tags.get("comment") != "blurred-v1":
        problems.append(f"{path}: comment tag is not blurred-v1")
    return problems


def repo_root():
    """Repository root above src/ (audit runs git there)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def audit_git():
    """No raw video, chunks, cache, masks or .env tracked (spec 09 4.7)."""
    problems = []
    proc = subprocess.run(["git", "ls-files"], cwd=repo_root(),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return [f"git ls-files failed: {proc.stderr.strip()}"]
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.endswith((".mov", ".MOV", ".mp4", ".MP4")):
            ok = ("/clips/" in line and os.path.basename(line).startswith("E-")
                  and line.endswith(".mp4")) or line.startswith("docs/clips/")
            if not ok:
                problems.append(f"tracked video outside clips dirs: {line}")
        if line.endswith(".npz") and not line.startswith("fixtures/"):
            problems.append(f"tracked npz outside fixtures: {line}")
    return problems


def audit(paths):
    """ffprobe every mp4, check jpg siblings, then the git tree."""
    problems = []
    clips_dirs = set()
    mp4s = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".mp4"):
                    mp4s.append(os.path.join(path, name))
            clips_dirs.add(path)
        else:
            mp4s.append(path)
            clips_dirs.add(os.path.dirname(path))
    passed = set()
    for mp4 in sorted(mp4s):
        file_problems = audit_file(mp4)
        problems.extend(file_problems)
        if not file_problems:
            passed.add(mp4)
        stem = os.path.splitext(mp4)[0]
        if not os.path.exists(stem + ".jpg"):
            problems.append(f"{mp4}: keyframe {stem}.jpg missing")
    for folder in sorted(clips_dirs):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".jpg") and os.path.join(
                    folder, os.path.splitext(name)[0] + ".mp4") not in passed:
                problems.append(f"{folder}/{name}: no passing sibling mp4")
    problems.extend(audit_git())
    return problems


def read_chapters(run_dir):
    """chunks.csv, tracks.csv, zones.json, events.json of one session."""
    with open(os.path.join(run_dir, "chunks.csv"), newline="") as fh:
        chunks = {r["chunk_id"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(run_dir, "tracks.csv"), newline="") as fh:
        tracks = list(csv.DictReader(fh))
    with open(os.path.join(run_dir, "zones.json"), encoding="utf-8") as fh:
        zones = json.load(fh)
    with open(os.path.join(run_dir, "events.json"), encoding="utf-8") as fh:
        events = json.load(fh)
    return chunks, tracks, zones, events


def chunk_path(run_dir, chunk_id):
    """Unblurred analysis chunk in the crossing cache (gitignored)."""
    crossing = os.path.dirname(os.path.abspath(run_dir))
    return os.path.join(crossing, "cache", "chunks", f"{chunk_id}.mp4")


def grab_frames(mp4, t0_s, a_s, b_s):
    """Decode window [a_s, b_s] at 10 fps; returns (frames, frame_times)."""
    import cv2

    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4}")
    try:
        f0 = max(0, int(round((a_s - t0_s) * FPS)))
        f1 = int(round((b_s - t0_s) * FPS))
        frames, times = [], []
        for f in range(f0, f1 + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"{mp4}: frame {f} unreadable")
            frames.append(img)
            times.append(round(t0_s + f / FPS, 2))
    finally:
        cap.release()
    return frames, times


def downscale720(img):
    """Height at most 720 px, never upscaled (for keyframes)."""
    import cv2

    h, w = img.shape[:2]
    if h <= HEIGHT:
        return img
    scale = HEIGHT / h
    return cv2.resize(img, (int(w * scale), HEIGHT),
                      interpolation=cv2.INTER_AREA)


MAX_GAP_S = 1.0     # tracking gaps up to this are interpolated for blur, so a missed detection stays blurred
HOLD_S = 0.2        # first/last box held this long past a track's ends


def cover(by_obj, t):
    """Boxes to blur at time t: observed, interpolated over a gap <= MAX_GAP_S, or held <= HOLD_S past a track end.

    Privacy first: a box held or interpolated a little too long only over-blurs.
    """
    out = []
    for obj_rows in by_obj.values():
        if not obj_rows[0]["_t"] - HOLD_S - 1e-9 <= t <= obj_rows[-1]["_t"] + HOLD_S + 1e-9:
            continue
        near = min(obj_rows, key=lambda r: abs(r["_t"] - t))
        if abs(near["_t"] - t) <= 0.5 / FPS + 1e-9 or t < obj_rows[0]["_t"] or t > obj_rows[-1]["_t"]:
            out.append(near)
            continue
        after = next(r for r in obj_rows if r["_t"] > t)
        before = max((r for r in obj_rows if r["_t"] < t), key=lambda r: r["_t"])
        if after["_t"] - before["_t"] > MAX_GAP_S + 1e-9:
            continue
        w = (t - before["_t"]) / (after["_t"] - before["_t"])
        box = {k: float(before[k]) + w * (float(after[k]) - float(before[k])) for k in ("x1", "y1", "x2", "y2")}
        out.append(dict(before, **box, _t=t))
    return out


def render_event(event, tracks, zones, run_dir, out_dir):
    """Blur, overlay, encode one event; returns (mp4, jpg, stats)."""
    import cv2

    chunk_id = event["chunk_id"]
    with open(os.path.join(run_dir, "chunks.csv"), newline="") as fh:
        meta = {r["chunk_id"]: r for r in csv.DictReader(fh)}[chunk_id]
    t0_s = float(meta["t0_s"])
    a_s, b_s = (float(v) for v in event["window_s"])
    ca_poly = next(a["polygon_px"] for a in zones["conflict_areas"]
                   if a["id"] == event["conflict_area"])
    rows = [r for r in tracks if r["chunk_id"] == chunk_id]
    for r in rows:
        r["_t"] = float(r["t_s"])
    first_id, second_id = event["first"]["obj_id"], event["second"]["obj_id"]
    frames, times = grab_frames(chunk_path(run_dir, chunk_id), t0_s, a_s, b_s)
    shape = frames[0].shape[:2]
    rendered = []
    persons_blurred = 0
    by_obj = {}
    for r in sorted(rows, key=lambda r: r["_t"]):
        by_obj.setdefault(r["obj_id"], []).append(r)
    for img, t in zip(frames, times):
        here = [r for r in rows if abs(r["_t"] - t) <= 0.5 / FPS + 1e-9]
        covered = cover(by_obj, t)                 # blur also spans short tracking gaps
        persons = [r for r in covered if r["cls"] == "person"]
        persons_blurred += len(persons)
        masks = [box_mask(shape, float(r["x1"]), float(r["y1"]),
                          float(r["x2"]), float(r["y2"])) for r in persons]
        # Spec 09 fallback: plates stayed readable at 720p with a partial box
        # blur (a high-mounted SUV plate), so every car box is blurred whole.
        plates = [box_mask(shape, float(r["x1"]), float(r["y1"]),
                           float(r["x2"]), float(r["y2"]))
                  for r in covered if r["cls"] == "car"]
        out = blur_frame(img, masks, plates)
        first = [r for r in here if r["obj_id"] == first_id]
        second = [r for r in here if r["obj_id"] == second_id]
        others = [r for r in here
                  if r["obj_id"] not in (first_id, second_id)]
        first_t = [r for r in rows if r["obj_id"] == first_id
                     and float(r["t_s"]) <= t]
        second_t = [r for r in rows if r["obj_id"] == second_id
                    and float(r["t_s"]) <= t]
        rendered.append(draw_legend(out, event, ca_poly, first, second,
                                    others, trail(first_t), trail(second_t)))
    mp4 = os.path.join(out_dir, f"{event['event_id']}.mp4")
    encode(rendered, FPS, mp4)
    key = min(range(len(times)),
              key=lambda i: abs(times[i] - event["second"]["t_enter_s"]))
    jpg = os.path.join(out_dir, f"{event['event_id']}.jpg")
    cv2.imwrite(jpg, downscale720(rendered[key]))
    return mp4, jpg, {"frames": len(rendered), "persons_blurred": persons_blurred}


def run_batch(run_dir, only=()):
    """Render every (or selected) event into the crossing clips/."""
    crossing = os.path.dirname(os.path.abspath(run_dir))
    out_dir = os.path.join(crossing, "clips")
    os.makedirs(out_dir, exist_ok=True)
    chunks, tracks, zones, events = read_chapters(run_dir)
    failures = []
    for event in events:
        if only and event["event_id"] not in only:
            continue
        try:
            mp4, jpg, stats = render_event(event, tracks, zones, run_dir,
                                           out_dir)
        except (RuntimeError, KeyError, ValueError) as exc:
            failures.append(f"{event['event_id']}: {exc}")
            continue
        print(f"{event['event_id']} frames={stats['frames']} "
              f"persons_blurred={stats['persons_blurred']} "
              f"clip={os.path.relpath(mp4, crossing)}")
    for failure in failures:
        print(f"FAILED: {failure}")
    return 1 if failures else 0


def check():
    """Synthetic demo: blur, masks, head box, crowd, window, encode, audit."""
    import cv2
    import numpy as np

    # gap cover: a person missing at t=0.2 between rows at 0.1 and 0.3 is still blurred, interpolated
    track = [{"obj_id": "p", "cls": "person", "_t": t, "x1": "0", "y1": "0", "x2": str(10 * k), "y2": "10"}
             for k, t in ((1, 0.1), (3, 0.3))]
    got = cover({"p": track}, 0.2)
    assert len(got) == 1 and abs(float(got[0]["x2"]) - 20.0) < 1e-6, got
    assert cover({"p": track}, 0.45) and not cover({"p": track}, 0.8)          # held 0.2 s, then gone
    edge = [dict(track[0], _t=1.2), dict(track[1], _t=2.2)]                   # exactly MAX_GAP_S apart
    assert cover({"p": edge}, 1.7), "a one-second gap must stay covered"
    far = [dict(track[0]), dict(track[1], _t=2.0)]
    assert not cover({"p": far}, 1.0)                                         # gap over MAX_GAP_S: not invented

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    person = np.zeros((1080, 1920), dtype=bool)
    person[400:520, 900:940] = True
    hair = np.zeros((1080, 1920), dtype=bool)
    hair[397:400, 900:906] = True
    plate = np.zeros((1080, 1920), dtype=bool)
    plate[700:710, 300:330] = True
    out = blur_frame(frame, [person | hair], [plate])
    region = dilate(person | hair) | head_box(person | hair) | dilate(plate)
    assert (out[~region] == frame[~region]).all()
    gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = lambda img: cv2.Laplacian(gray(img), cv2.CV_64F)[region]
    before, after = lap(frame).var(), lap(out).var()
    assert after <= 0.1 * before, (before, after)
    assert (out[397:400, 900:906] != frame[397:400, 900:906]).any()
    crowded = blur_frame(frame, [person] * MAX_IDS, [])
    assert not (crowded == frame).all()
    outside = np.zeros((1080, 1920), dtype=bool)
    outside[0:10, 0:10] = True
    same = blur_frame(frame, [outside], [])
    small_region = dilate(outside) | head_box(outside)
    assert (same[~small_region] == frame[~small_region]).all()
    assert (same[small_region] != frame[small_region]).any()
    assert clip_window(10.0, 12.9) == [8.0, 14.9]
    assert clip_window(10.0, 8.0) == [6.0, 12.0]
    with tempfile.TemporaryDirectory() as tmp:
        small = [cv2.resize(frame, (320, 240),
                            interpolation=cv2.INTER_AREA) for _ in range(10)]
        good = os.path.join(tmp, "E-0001.mp4")
        encode(small, FPS, good)
        jpg = os.path.join(tmp, "E-0001.jpg")
        cv2.imwrite(jpg, downscale720(small[0]))
        assert audit([good]) == [], audit([good])
        bad = os.path.join(tmp, "E-0002.mp4")
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", good,
               "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
               "-shortest", "-c:v", "copy", "-c:a", "aac",
               "-metadata", "location=+25.6537-100.2877/", bad]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr.strip()
        problems = audit([bad])
        assert any("audio" in p for p in problems), problems
        assert any("location" in p for p in problems), problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evidence clips (spec 09).")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("events", nargs="*")
    parser.add_argument("--window", nargs=3, default=None, metavar=("CHUNK", "T0", "T1"),
                        help="review render (not implemented)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--concept", default=None)
    parser.add_argument("--audit", nargs="+", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        print("clips --check: PASS (blur, masks, head, crowd, window, encode, audit)")
        return
    if args.audit is not None:
        problems = audit(args.audit)
        for problem in problems:
            print(f"AUDIT: {problem}")
        raise SystemExit(1 if problems else 0)
    if args.window or args.out or args.concept:
        parser.error("review render and concept render are not implemented")
    if not args.run_dir:
        parser.error("need RUN_DIR (or --check / --audit PATHS)")
    try:
        code = run_batch(args.run_dir, args.events)
    except (RuntimeError, KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
