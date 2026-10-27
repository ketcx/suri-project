"""Session ingest: raw video to 10 fps analysis chunks.

One ffmpeg call per chunk, ROI-cropped, no audio, H.264 faststart.
Writes ``chunks.csv`` plus ``runs/<crossing>/cache/chunks/<id>.mp4``.

The runs tree root comes from ``SURI_RUNS_DIR`` (default ``runs``) so the
self-check can run in a temporary directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

FPS = 10

CHUNKS_COLUMNS = (
    "chunk_id",
    "session_id",
    "src_file",
    "t0_s",
    "duration_s",
    "fps",
    "width",
    "height",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "sha256",
)


def runs_dir() -> str:
    return os.environ.get("SURI_RUNS_DIR", "runs")


def plan_chunks(durations, chunk_s=300.0, overlap_s=1.0):
    """Plan (src_idx, start_s, dur_s, t0_s) per chunk. No chunk spans files."""
    out, offset = [], 0.0
    for i, total in enumerate(durations):
        start = 0.0
        while True:
            out.append((i, start, min(chunk_s, total - start), offset + start))
            if start + chunk_s >= total:
                break
            start += chunk_s - overlap_s
        offset += total
    return out


def parse_roi(text: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise ValueError(f"ROI must be X,Y,W,H, got {text!r}") from None
    if min(x, y) < 0 or min(w, h) <= 0:
        raise ValueError(f"ROI must be non-empty with x,y >= 0, got {text!r}")
    if w % 4 or h % 4:
        raise ValueError(f"ROI w,h must be multiples of 4, got {text!r}")
    return x, y, w, h


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}… failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def probe_duration(path: str) -> float:
    """Duration of the video stream, not the container (audio may run long)."""
    out = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ]
    )
    doc = json.loads(out)
    streams = doc.get("streams") or []
    if streams and streams[0].get("duration"):
        return float(streams[0]["duration"])
    return float(doc["format"]["duration"])


def probe_chunk(path: str) -> dict:
    out = _run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries", "stream=codec_name,r_frame_rate,width,height,nb_read_frames",
            "-show_entries", "format=nb_streams",
            "-of", "json", path,
        ]
    )
    doc = json.loads(out)
    return {"streams": doc["format"]["nb_streams"], **doc["streams"][0]}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_chunk(path: str, width: int, height: int, n_frames: int) -> None:
    """ffprobe gate: 10 fps H.264, one stream, exact frame count and size."""
    info = probe_chunk(path)
    if info["streams"] != 1:
        raise ValueError(f"{path}: expected 1 stream, got {info['streams']}")
    if info["codec_name"] != "h264":
        raise ValueError(f"{path}: expected h264, got {info['codec_name']}")
    if info["r_frame_rate"] != "10/1":
        raise ValueError(f"{path}: expected 10/1 fps, got {info['r_frame_rate']}")
    if int(info["width"]) != width or int(info["height"]) != height:
        raise ValueError(f"{path}: expected {width}x{height}, got {info['width']}x{info['height']}")
    if int(info["nb_read_frames"]) != n_frames:
        raise ValueError(f"{path}: expected {n_frames} frames, got {info['nb_read_frames']}")


def verify_coverage(plan, durations, chunk_s=300.0, overlap_s=1.0, tol=0.1) -> None:
    """Coverage rule: first t0 is 0, 1 s overlaps inside a file, files touch."""
    if not plan:
        raise ValueError("empty chunk plan")
    if abs(plan[0][3]) > 1e-6:
        raise ValueError(f"first t0_s is {plan[0][3]}, expected 0")
    for prev, cur in zip(plan, plan[1:]):
        gap = cur[3] - (prev[3] + prev[2])
        if cur[0] == prev[0]:
            if abs(gap + overlap_s) > tol:
                raise ValueError(f"overlap is {-gap:.3f}s, expected {overlap_s}s")
        elif abs(gap) > tol:
            raise ValueError(f"file boundary gap is {gap:.3f}s, expected ~0")
    total = sum(durations)
    end = plan[-1][3] + plan[-1][2]
    if abs(end - total) > tol:
        raise ValueError(f"plan ends at {end:.3f}s, session is {total:.3f}s")


def encode_chunk(src: str, start: float, dur: float, roi, half: bool, out_path: str) -> None:
    x, y, w, h = roi
    n_frames = round(dur * FPS)
    vf = f"fps={FPS},crop={w}:{h}:{x}:{y}"
    width, height = w, h
    if half:
        vf += ",scale=iw/2:ih/2"
        width, height = w // 2, h // 2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
            "-vf", vf, "-frames:v", str(n_frames),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ]
    )
    verify_chunk(out_path, width, height, n_frames)


def ingest(crossing, session, roi, raw_files, chunk_s=300.0, overlap_s=1.0, half=False):
    """Encode the session and write chunks.csv. Returns the row dicts."""
    x, y, w, h = roi
    width, height = (w // 2, h // 2) if half else (w, h)
    session_dir = os.path.join(runs_dir(), crossing, session)
    cache_dir = os.path.join(runs_dir(), crossing, "cache", "chunks")
    csv_path = os.path.join(session_dir, "chunks.csv")
    recorded = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                recorded[row["chunk_id"]] = row
    durations = [probe_duration(p) for p in raw_files]
    plan = plan_chunks(durations, chunk_s, overlap_s)
    verify_coverage(plan, durations, chunk_s, overlap_s)
    rows = []
    for n, (src_idx, start, dur, t0) in enumerate(plan):
        chunk_id = f"{session}-c{n:02d}"
        out_path = os.path.join(cache_dir, chunk_id + ".mp4")
        n_frames = round(dur * FPS)
        planned = {
            "chunk_id": chunk_id,
            "session_id": session,
            "src_file": os.path.basename(raw_files[src_idx]),
            "t0_s": round(t0, 3),
            "duration_s": round(n_frames / FPS, 3),
            "fps": FPS,
            "width": width,
            "height": height,
            "roi_x": x,
            "roi_y": y,
            "roi_w": w,
            "roi_h": h,
        }
        # Reuse only when every planned field matches and the file is intact:
        # a re-ingest with another ROI, --half or --chunk-s must re-encode.
        old = recorded.get(chunk_id)
        if (
            old is not None
            and all(str(old.get(k)) == str(v) for k, v in planned.items())
            and os.path.exists(out_path)
            and old.get("sha256") == sha256_file(out_path)
        ):
            digest = old["sha256"]
        else:
            encode_chunk(raw_files[src_idx], start, dur, roi, half, out_path)
            digest = sha256_file(out_path)
        rows.append({**planned, "sha256": digest})
    os.makedirs(session_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CHUNKS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _check() -> None:
    """Ingest self-check: 65 s testsrc2, re-runs, param changes, long audio."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="suri-ingest-check-")
    os.environ["SURI_RUNS_DIR"] = os.path.join(tmp, "runs")
    try:
        raw = os.path.join(tmp, "raw.mp4")
        _run(
            [
                "ffmpeg", "-nostdin", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=65",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=65",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", raw,
            ]
        )
        rows = ingest("x", "s1", (0, 0, 320, 240), [raw], chunk_s=30.0)
        assert [r["chunk_id"] for r in rows] == ["s1-c00", "s1-c01", "s1-c02"], rows
        assert [r["t0_s"] for r in rows] == [0.0, 29.0, 58.0], rows
        frames = []
        for row in rows:
            path = os.path.join(tmp, "runs", "x", "cache", "chunks", row["chunk_id"] + ".mp4")
            info = probe_chunk(path)
            assert info["streams"] == 1 and info["r_frame_rate"] == "10/1", info
            assert row["sha256"] == sha256_file(path), row["chunk_id"]
            frames.append(int(info["nb_read_frames"]))
        assert frames == [300, 300, 70], frames
        stamp = {r["chunk_id"]: os.path.getmtime(os.path.join(tmp, "runs", "x", "cache", "chunks", r["chunk_id"] + ".mp4")) for r in rows}
        rows2 = ingest("x", "s1", (0, 0, 320, 240), [raw], chunk_s=30.0)
        assert rows2 == rows, "second run must return identical rows"
        for r in rows:
            path = os.path.join(tmp, "runs", "x", "cache", "chunks", r["chunk_id"] + ".mp4")
            assert os.path.getmtime(path) == stamp[r["chunk_id"]], f"re-encoded {r['chunk_id']}"

        # A re-ingest with other params must re-encode, not reuse stale chunks.
        sha_before = {r["chunk_id"]: r["sha256"] for r in rows}
        rows3 = ingest("x", "s1", (0, 0, 320, 240), [raw], chunk_s=20.0)
        assert [r["chunk_id"] for r in rows3] == ["s1-c00", "s1-c01", "s1-c02", "s1-c03"], rows3
        assert [r["t0_s"] for r in rows3] == [0.0, 19.0, 38.0, 57.0], rows3
        assert rows3[0]["sha256"] != sha_before["s1-c00"], "stale 30 s chunk reused"
        rows4 = ingest("x", "s1", (0, 0, 160, 120), [raw], chunk_s=20.0)
        assert (rows4[0]["width"], rows4[0]["height"]) == (160, 120), rows4[0]
        assert rows4[0]["sha256"] != rows3[0]["sha256"], "stale full-ROI chunk reused"

        # duration_s follows frames/fps, and audio running long does not stop us.
        long_audio = os.path.join(tmp, "raw-long-audio.mp4")
        _run(
            [
                "ffmpeg", "-nostdin", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=65",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=66",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", long_audio,
            ]
        )
        rows5 = ingest("x", "s2", (0, 0, 320, 240), [long_audio], chunk_s=30.0)
        total_frames = sum(int(r["duration_s"] * 10) for r in rows5)
        assert total_frames == 670, rows5
        assert rows5[-1]["duration_s"] == 7.0, rows5[-1]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ingest --check: PASS (chunks, idempotent, re-ingest, long audio)")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw session video into 10 fps chunks.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--crossing", default="mty-garciaroel-ahuehuetes")
    parser.add_argument("--session", default="")
    parser.add_argument("--roi", default="")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--chunk-s", type=float, default=300.0)
    parser.add_argument("--overlap-s", type=float, default=1.0)
    parser.add_argument("raw", nargs="*")
    args = parser.parse_args(argv)
    if args.check:
        _check()
        return
    if not args.session or not args.roi or not args.raw:
        parser.error("--session, --roi X,Y,W,H and at least one RAW file are required")
    ingest(args.crossing, args.session, parse_roi(args.roi), args.raw, args.chunk_s, args.overlap_s, args.half)


if __name__ == "__main__":
    main()
