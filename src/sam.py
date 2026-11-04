"""SAM 3.1 segmentation client: chunk windows to tracks and masks.

One request per concept (``person``, then ``car``) per chunk through
``ledger.call``. Media enters the request only as a chunk sha256 key, so a
re-cut still hits the cache. See spec 02 for the wire contract.

Layout notes (spec 02 section 3, PRD section 7):

- ``tracks.csv`` columns are exactly ``TRACKS_COLUMNS``. ``obj_id`` is
  ``<chunk_id>:<cls>:<sam_id>``; ``t_s`` is session time with 3 decimals;
  ``X_m``/``Y_m`` stay empty for kinematics.
- ``masks/<chunk_id>.npz`` holds one record per object and frame with the
  keys ``obj_id, frame, mask_hw, crop, offset, bits, frame_wh``.
- Uploads use the Files API with ``purpose=user_data`` and get no ledger
  row. One upload per chunk serves both concepts within a batch run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import ledger

MODEL = "sam-3.1"
ENDPOINT = "meta/responses"
BASE_URL = "https://api.meta.ai/v1"
FPS = 10
VARIANT = "raw"
ESTIMATE_PER_1000 = 0.20
MIN_BILLED_FRAMES = 50
MAX_OBJECTS_PER_FRAME = 16

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

CONCEPT_RE = re.compile(r"^[a-z][a-z ]{1,40}$")
FRAME_RE = re.compile(r"<(\d+)f>")
SPLIT_RE = re.compile(r"(?<=\|>),(?=\d+<\|box;)")
REC_RE = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;(.*)\|>", re.S)


class ParseError(Exception):
    """A stream line that does not match the section 4.3 grammar."""


def check_concept(concept):
    """Concepts match ^[a-z][a-z ]{1,40}$ so obj_id can split on ':'."""
    if not isinstance(concept, str) or not CONCEPT_RE.match(concept):
        raise ValueError(f"bad concept {concept!r}")
    return concept


def estimate_usd(n_frames):
    """Spec 02 section 4.6: max(50, frames) * 0.20 / 1000."""
    return max(MIN_BILLED_FRAMES, int(n_frames)) * ESTIMATE_PER_1000 / 1000.0


def build_request_body(concept, file_ref):
    """The exact section 4.2 body; file_ref is a file_id or a media key."""
    return {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": check_concept(concept)},
                    {"type": "input_video", "file_id": file_ref},
                ],
            }
        ],
        "stream": True,
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    }


def build_key_body(concept, chunk_sha, n_frames, variant=VARIANT):
    """Hashable body: the request with the media reference as a media key."""
    return {
        "concept": check_concept(concept),
        "chunk_sha": chunk_sha,
        "frames": int(n_frames),
        "variant": variant,
        "body": build_request_body(concept, f"media:{chunk_sha}"),
    }


def get_client():
    """OpenAI client on the Meta base URL with MODEL_API_KEY explicit."""
    try:
        import openai
    except ImportError:
        raise RuntimeError(
            "the openai package is required for real calls; "
            "install it with: pip install openai"
        ) from None
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set")
    return openai.OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0)


def upload_chunk(client, path):
    """Upload one chunk file; returns the file_id. No ledger row."""
    with open(path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="user_data")
    file_id = getattr(uploaded, "id", None)
    if not file_id:
        raise ValueError(f"upload of {path} returned no file id")
    return file_id


def send_request(client, file_id, concept):
    """POST one streaming request; returns the raw event dicts in order."""
    body = build_request_body(concept, file_id)
    stream = client.responses.create(**body)
    return [event.model_dump() for event in stream]


def decode_mask(payload, mh, mw):
    """Decode a one_bit payload: base85, unpack bits, first MH*MW row-major.

    Tries RFC 1924 then Adobe Ascii85. The live streams use an
    undocumented variable-length alphabet (no , | or ~, so record
    splitting stays safe); those payloads raise ParseError and the
    caller keeps the row with empty bits instead of dropping it.
    """
    import numpy as np

    raw = None
    for decode in (base64.b85decode,
                   lambda body: base64.a85decode(body, adobe=False)):
        try:
            raw = decode(payload)
            break
        except Exception:
            continue
    if raw is None:
        raise ParseError("one_bit payload is not RFC1924 or Ascii85 base85")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    if bits.size < mh * mw:
        raise ParseError(f"mask payload too short for {mh}x{mw}")
    return bits[: mh * mw].reshape((mh, mw)).astype(bool)


def ground_point(mask, frame_wh):
    """Midpoint of the lowest 5% of mask rows, in frame pixels (4.4).

    Kept for the full-raster path; parse_events uses the box bottom
    centre until the one_bit crop mapping of bug 2 is wired in.
    """
    import numpy as np

    width, height = frame_wh
    rows = np.flatnonzero(mask.any(1))
    if len(rows) == 0:
        raise ValueError("empty mask has no ground point")
    r1 = int(rows[-1])
    k = max(1, (r1 - int(rows[0]) + 1 + 19) // 20)
    cols = np.flatnonzero(mask[r1 - k + 1: r1 + 1].any(0))
    mh, mw = mask.shape
    gx = (int(cols[0]) + int(cols[-1]) + 1) / 2 * width / mw
    gy = (2 * r1 - k + 2) / 2 * height / mh
    return float(gx), float(gy)


def parse_events(events, chunk, cls):
    """Parse cached or live SSE events into tracks rows and mask crops.

    Returns (rows, crops, stats) where rows follow TRACKS_COLUMNS and
    crops holds one mask record per object and frame for the npz layout.
    Raises ParseError with the line number on any grammar violation.
    """
    import numpy as np

    width, height, n_frames = chunk["width"], chunk["height"], chunk["n_frames"]
    texts = []
    dropped = 0
    for event in events:
        if not isinstance(event, dict):
            raise ParseError("event is not a dict")
        if event.get("type") != "response.output_text.delta":
            continue  # created/in_progress/added/done/completed carry no rows
        for line in (event.get("delta") or "").split("\n"):
            if line.strip():
                texts.append(line)
    records = []
    for lineno, line in enumerate(texts):
        match = FRAME_RE.match(line)
        if not match or int(match[1]) >= n_frames:
            raise ParseError(f"line {lineno}: bad frame header")
        frame = int(match[1])
        for part in SPLIT_RE.split(line[match.end():]):
            rec = REC_RE.fullmatch(part)
            if not rec:
                raise ParseError(f"line {lineno}: bad record")
            sid_s, box_s, mask_s = rec.groups()
            box = dict(kv.split("=", 1) for kv in box_s.split(";") if kv)
            try:
                x1, y1 = float(box["x1"]), float(box["y1"])
                x2, y2 = float(box["x2"]), float(box["y2"])
                bw, bh = float(box["w"]), float(box["h"])
            except (KeyError, ValueError):
                raise ParseError(f"line {lineno}: bad box coords") from None
            if bw <= 0 or bh <= 0:
                raise ParseError(f"line {lineno}: bad box size")
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                dropped += 1
                continue
            head, _, data = mask_s.partition("data=")
            mh_s, _, payload = data.partition(",")
            mw_s, _, payload = payload.partition(",")
            try:
                mh, mw = int(mh_s), int(mw_s)
            except ValueError:
                raise ParseError(f"line {lineno}: bad mask size") from None
            marker, b85 = payload[0:1], payload[1:]
            if marker != "!":
                raise ParseError(f"line {lineno}: want one_bit, got {marker!r}")
            meta = dict(kv.split("=", 1) for kv in head.split(";") if kv and "=" in kv)
            if meta.get("x", "0") != "0" or meta.get("y", "0") != "0":
                raise ParseError(f"line {lineno}: mask offset must be x=0;y=0")
            records.append(
                {
                    "frame": frame,
                    "sam_id": str(int(sid_s)),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "bw": bw,
                    "bh": bh,
                    "mask_c": meta.get("c", ""),
                    "box_c": box.get("c", ""),
                    "mh": mh,
                    "mw": mw,
                    "b85": b85,
                    "lineno": lineno,
                }
            )
    records.sort(key=lambda r: (r["frame"], int(r["sam_id"])))
    rows, crops, seen = [], [], set()
    stats = {
        "records": len(records),
        "dropped": dropped,
        "dup": 0,
        "empty_masks": 0,
        "undecoded_masks": 0,
        "scaled": False,
        "ids": set(),
        "frames_with_objects": set(),
        "per_frame": {},
    }
    for rec in records:
        key = (rec["frame"], rec["sam_id"])
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        x1, y1, x2, y2 = rec["x1"], rec["y1"], rec["x2"], rec["y2"]
        if rec["bw"] != width or rec["bh"] != height:
            sx, sy = width / rec["bw"], height / rec["bh"]
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            stats["scaled"] = True
        if not (-1 <= x1 < x2 <= width + 1 and -1 <= y1 < y2 <= height + 1):
            raise ParseError(f"line {rec['lineno']}: box outside the frame")
        try:
            mask = decode_mask(rec["b85"], rec["mh"], rec["mw"])
        except ParseError:
            mask = None
        if mask is None:
            stats["undecoded_masks"] += 1
        elif not mask.any():
            stats["empty_masks"] += 1
        gx, gy = (x1 + x2) / 2, y2  # box bottom centre (today's simplification)
        if mask is None:
            r0, r1, c0, c1, bits = 0, 0, 0, 0, np.packbits(np.zeros(0, dtype=bool))
        else:
            r0, r1, c0, c1 = _crop_of(mask)
            bits = np.packbits(mask[r0:r1, c0:c1].reshape(-1))
        crops.append(
            {
                "obj_id": f"{chunk['chunk_id']}:{cls}:{rec['sam_id']}",
                "frame": rec["frame"],
                "mask_hw": (rec["mh"], rec["mw"]),
                "crop": (r0, r1, c0, c1),
                "bits": bits,
            }
        )
        t_s = round(chunk["t0_s"] + rec["frame"] / FPS, 3)
        rows.append(
            {
                "session_id": chunk["session_id"],
                "chunk_id": chunk["chunk_id"],
                "frame": rec["frame"],
                "t_s": t_s,
                "obj_id": f"{chunk['chunk_id']}:{cls}:{rec['sam_id']}",
                "cls": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "gx_px": gx,
                "gy_px": gy,
                "X_m": "",
                "Y_m": "",
                "conf": rec["box_c"] or rec["mask_c"],
            }
        )
        stats["ids"].add(rec["sam_id"])
        stats["frames_with_objects"].add(rec["frame"])
        stats["per_frame"][rec["frame"]] = stats["per_frame"].get(rec["frame"], 0) + 1
    peak = max(stats["per_frame"].values(), default=0)
    stats["peak_ids"] = peak
    stats["frames_at_cap"] = sum(1 for n in stats["per_frame"].values() if n >= MAX_OBJECTS_PER_FRAME)
    return rows, crops, stats


def _crop_of(mask):
    """End-exclusive raster crop of the set pixels; empty mask -> (0,0,0,0)."""
    import numpy as np

    rows = np.flatnonzero(mask.any(1))
    if len(rows) == 0:
        return (0, 0, 0, 0)
    cols = np.flatnonzero(mask.any(0))
    return (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def write_masks_npz(path, crops, frame_wh):
    """Write one chunk's mask records in the section 3 npz layout."""
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj_ids = [c["obj_id"] for c in crops]
    frames = np.array([c["frame"] for c in crops], dtype=np.int32)
    hw = np.array([c["mask_hw"] for c in crops], dtype=np.int32)
    box = np.array([c["crop"] for c in crops], dtype=np.int32)
    concat = (
        np.concatenate([c["bits"] for c in crops])
        if crops
        else np.array([], dtype=np.uint8)
    )
    offset = np.zeros(len(crops) + 1, dtype=np.int64)
    for i, c in enumerate(crops):
        offset[i + 1] = offset[i] + len(c["bits"])
    np.savez_compressed(
        path,
        obj_id=np.array(obj_ids),
        frame=frames,
        mask_hw=hw,
        crop=box,
        offset=offset,
        bits=concat,
        frame_wh=np.array(list(frame_wh), dtype=np.int32),
    )


def load_mask(npz_path, i):
    """The (MH, MW) bool raster of record i, as stored in section 3."""
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as data:
        crop = data["crop"][i]
        offset = data["offset"]
        bits = data["bits"][offset[i]: offset[i + 1]]
        r0, r1, c0, c1 = (int(v) for v in crop)
        mh, mw = (int(v) for v in data["mask_hw"][i])
        raster = np.zeros((mh, mw), dtype=bool)
        if r1 > r0 and c1 > c0:
            flat = np.unpackbits(bits)[: (r1 - r0) * (c1 - c0)]
            raster[r0:r1, c0:c1] = flat.reshape((r1 - r0, c1 - c0)).astype(bool)
    return raster


def parse_selector(selector, chunks):
    """Split CONCEPT[:CHUNKS] into (concept, [chunk dicts])."""
    concept, _, which = selector.partition(":")
    check_concept(concept)
    if not which or which == "all":
        return concept, list(chunks)
    wanted = set()
    for part in which.split(","):
        if "-" in part:
            head, tail = part.split("-", 1)
            prefix = head.rstrip("0123456789")
            lo_s, hi_s = head[len(prefix):], tail
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                wanted.update(f"{prefix}{i:02d}" for i in range(lo, hi + 1))
            else:
                wanted.add(part)
        else:
            wanted.add(part)
    have = {c["chunk_id"].rsplit("-", 1)[-1] for c in chunks}
    have |= {c["chunk_id"] for c in chunks}
    if wanted - have:
        raise ValueError(f"unknown chunks: {sorted(wanted - have)}")
    return concept, [c for c in chunks
                     if c["chunk_id"] in wanted
                     or c["chunk_id"].rsplit("-", 1)[-1] in wanted]


def read_chunks(run_dir):
    """Read and validate a session chunks.csv."""
    path = os.path.join(run_dir, "chunks.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if abs(float(row["fps"]) - FPS) > 1e-9:
            raise ValueError(f"{row['chunk_id']}: fps must be 10")
    return rows


def chunk_mp4_path(run_dir, chunk_id):
    """Unblurred chunk cache, gitignored (spec 02 section 3)."""
    crossing_dir = os.path.dirname(os.path.abspath(run_dir))
    return os.path.join(crossing_dir, "cache", "chunks", f"{chunk_id}.mp4")


def cut_window(src_mp4, f0, f1, dst_mp4):
    """Cut frames [f0, f1) with ffmpeg; returns dst_mp4."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", str(f0 / FPS),
        "-i", src_mp4, "-frames:v", str(f1 - f0),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", dst_mp4,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst_mp4):
        raise RuntimeError(f"window cut failed: {proc.stderr.strip()}")
    return dst_mp4


def overlay_frame(chunk_mp4, frame_no, rows, out_png):
    """Draw boxes with obj_id on one frame; returns out_png."""
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        still = os.path.join(tmp, "frame.png")
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-ss", str(frame_no / FPS),
            "-i", chunk_mp4, "-frames:v", "1", still,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"frame extract failed: {proc.stderr.strip()}")
        img = cv2.imread(still)
    if img is None:
        raise RuntimeError(f"could not read extracted frame {frame_no}")
    for row in rows:
        if int(row["frame"]) != frame_no:
            continue
        x1, y1, x2, y2 = (int(float(row[k])) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(row["obj_id"]), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    if not cv2.imwrite(out_png, img):
        raise ValueError(f"could not write {out_png}")
    return out_png


def run_batch(run_dir, selectors, task_id, window=None):
    """Run every concept/chunk pair; write tracks.csv and masks. Returns 0/1."""
    import time

    chunks = read_chunks(run_dir)
    session_id = os.path.basename(os.path.abspath(run_dir))
    pairs = []
    for selector in selectors or ["person:all", "car:all"]:
        concept, picked = parse_selector(selector, chunks)
        pairs.extend((concept, c) for c in picked)
    if not pairs:
        raise ValueError("no chunk/concept pairs selected")
    clients = {}
    file_ids = {}
    all_rows, failures = [], []
    crops_by_chunk = {}
    total_cost, total_frames = 0.0, 0
    total_seconds = sum(float(c["duration_s"]) for c in chunks)
    for concept, chunk in pairs:
        chunk_id = chunk["chunk_id"]
        n_frames = int(round(float(chunk["duration_s"]) * FPS))
        src_mp4 = chunk_mp4_path(run_dir, chunk_id)
        if not os.path.exists(src_mp4):
            failures.append(f"{chunk_id}: missing {src_mp4}")
            continue
        media_video, media_key = src_mp4, chunk["sha256"]
        tmp_dir = None
        if window is not None:
            f0, f1 = window
            tmp_dir = tempfile.mkdtemp(prefix="sam-window-")
            media_video = cut_window(
                src_mp4, f0, f1, os.path.join(tmp_dir, f"{chunk_id}.mp4"))
            media_key = f"{chunk['sha256']}:{f0}-{f1}:{VARIANT}"
            n_frames = f1 - f0
        key_body = build_key_body(concept, media_key, n_frames)
        est = estimate_usd(n_frames)

        def send(file_ids=file_ids, clients=clients,
                 media_video=media_video, chunk_sha=chunk["sha256"],
                 concept=concept):
            if "client" not in clients:
                clients["client"] = get_client()
            client = clients["client"]
            fid = file_ids.get(chunk_sha)
            if fid is None:
                fid = upload_chunk(client, media_video)
                file_ids[chunk_sha] = fid
            return send_request(client, fid, concept)

        started = time.monotonic()
        try:
            events = list(ledger.call(
                task_id, MODEL, ENDPOINT, key_body, send,
                estimate_usd=est, cache_reads=True))
        except Exception as exc:
            failures.append(f"{chunk_id}/{concept}: {type(exc).__name__}: {exc}")
            continue
        finally:
            if tmp_dir is not None:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
        latency = time.monotonic() - started
        info = dict(chunk)
        info.update(
            chunk_id=chunk_id,
            session_id=chunk.get("session_id") or session_id,
            t0_s=float(chunk["t0_s"]) + (window[0] / FPS if window else 0.0),
            n_frames=n_frames,
            width=int(chunk["width"]),
            height=int(chunk["height"]),
        )
        try:
            rows, crops, stats = parse_events(events, info, concept)
        except ParseError as exc:
            failures.append(f"{chunk_id}/{concept}: ParseError: {exc}")
            continue
        usage_frames = 0
        for event in events:
            if isinstance(event, dict) and event.get("type") == "response.completed":
                usage = event.get("usage") or {}
                usage_frames = int(usage.get("video_frames_processed", 0))
                break
        billed = max(MIN_BILLED_FRAMES, usage_frames or n_frames)
        cost = billed * ESTIMATE_PER_1000 / 1000.0
        total_cost += cost
        total_frames += billed
        all_rows.extend(rows)
        crops_by_chunk.setdefault(chunk_id, []).extend(crops)
        n_with = len(stats["frames_with_objects"])
        saturated = stats["frames_at_cap"] / max(n_with, 1) > 0.01
        print(
            f"{chunk_id} {concept}: rows={len(rows)} ids={len(stats['ids'])} "
            f"peak_ids={stats['peak_ids']} frames_at_cap={stats['frames_at_cap']}"
            f"{' SATURATED' if saturated else ''} "
            f"empty_masks={stats['empty_masks']} "
            f"undecoded_masks={stats['undecoded_masks']} "
            f"dropped={stats['dropped']} dup={stats['dup']} "
            f"latency={latency:.1f}s billed={billed} USD {cost:.4f}")
    for failure in failures:
        print(f"FAILED: {failure}")
    if failures:
        print(f"total: not written rows={len(all_rows)} failures={len(failures)}")
        return 1
    order = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    all_rows.sort(key=lambda r: (order.get(r["chunk_id"], 0), r["frame"],
                                 r["cls"], r["obj_id"]))
    tracks_path = os.path.join(run_dir, "tracks.csv")
    with open(tracks_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACKS_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    masks_dir = os.path.join(run_dir, "masks")
    dims = {c["chunk_id"]: (int(c["width"]), int(c["height"])) for c in chunks}
    for chunk_id, crops in crops_by_chunk.items():
        write_masks_npz(os.path.join(masks_dir, f"{chunk_id}.npz"), crops,
                        dims[chunk_id])
    minutes = total_seconds / 60.0
    print(f"total: chunks={len(crops_by_chunk)} rows={len(all_rows)} "
          f"billed_frames={total_frames} USD {total_cost:.4f} "
          f"USD/min {total_cost / minutes if minutes else 0:.4f}")
    return 0


def _synthetic_line(frame, records, width=320, height=240):
    """Build one grammar-conformant delta line from record dicts."""
    import numpy as np

    parts = []
    for rec in records:
        mask = rec["mask"]
        mh, mw = mask.shape
        payload = base64.b85encode(np.packbits(mask.reshape(-1))).decode("ascii")
        box = (f"box;x1={rec['x1']};y1={rec['y1']};x2={rec['x2']};y2={rec['y2']}"
               f";w={width};h={height}")
        if rec.get("c"):
            box += f";c={rec['c']}"
        head = "x=0;y=0;"
        if rec.get("mc"):
            head += f"c={rec['mc']};"
        parts.append(f"{rec['sid']}<|{box}|><|mask;{head}data={mh},{mw},!{payload}|>")
    return f"<{frame}f>" + ",".join(parts)


def check():
    """Offline check on synthetic grammar-conformant lines (no key, no net)."""
    import numpy as np

    chunk = {"chunk_id": "g1-20261104-c00", "session_id": "g1-20261104",
             "t0_s": 0.0, "n_frames": 100, "width": 320, "height": 240}
    full = np.zeros((100, 100), dtype=bool)
    full[20:60, 10:20] = True
    small = np.zeros((40, 40), dtype=bool)
    small[5:25, 5:15] = True
    empty = np.zeros((30, 30), dtype=bool)
    line0 = _synthetic_line(0, [
        {"sid": 3, "x1": 10, "y1": 20, "x2": 50, "y2": 90, "mask": small, "c": "0.9"},
        {"sid": 7, "x1": 100, "y1": 100, "x2": 200, "y2": 200, "mask": full, "mc": "0.5"},
    ])
    line1 = _synthetic_line(1, [
        {"sid": 3, "x1": 12, "y1": 20, "x2": 52, "y2": 90, "mask": small},
        {"sid": 3, "x1": 12, "y1": 20, "x2": 52, "y2": 90, "mask": small},
        {"sid": 9, "x1": 0, "y1": 0, "x2": 30, "y2": 30, "mask": empty},
    ])
    noisy = np.random.default_rng(7).random((24, 24)) < 0.3
    probe = base64.b85encode(np.packbits(noisy.reshape(-1))).decode("ascii")
    assert "!" in probe[1:]  # a bang inside the payload is legal
    line2 = _synthetic_line(2, [
        {"sid": 5, "x1": 200, "y1": 150, "x2": 260, "y2": 210, "mask": noisy},
    ])
    blob = lambda line: {"type": "response.output_text.delta", "delta": line}
    events = [blob(line0), blob(line1), blob(line2),
              {"type": "response.output_text.done", "text": "\n".join([line0, line1, line2])},
              {"type": "response.completed",
               "usage": {"video_frames_processed": 100}}]
    rows, crops, stats = parse_events(events, chunk, "person")
    assert len(rows) == 5, len(rows)  # dup kept once; done blob ignored
    assert stats["dup"] == 1
    assert stats["empty_masks"] == 1
    assert stats["undecoded_masks"] == 0
    assert {r["obj_id"] for r in rows} == {
        "g1-20261104-c00:person:3", "g1-20261104-c00:person:7",
        "g1-20261104-c00:person:9", "g1-20261104-c00:person:5"}
    assert stats["peak_ids"] == 2
    assert all(list(TRACKS_COLUMNS) == list(r.keys()) for r in rows)
    assert all(r["X_m"] == "" and r["Y_m"] == "" for r in rows)
    assert all(r["t_s"] == round(r["frame"] / 10.0, 3) for r in rows)
    assert all(r["gx_px"] == (r["x1"] + r["x2"]) / 2 for r in rows)
    assert all(r["gy_px"] == r["y2"] for r in rows)
    assert (rows[0]["gx_px"], rows[0]["gy_px"]) == (30.0, 90.0)
    assert (rows[1]["gx_px"], rows[1]["gy_px"]) == (150.0, 200.0)
    assert (rows[4]["gx_px"], rows[4]["gy_px"]) == (230.0, 210.0)
    fallen = [r for r in rows if r["obj_id"].endswith(":9")][0]
    assert (fallen["gx_px"], fallen["gy_px"]) == (15.0, 30.0)
    assert rows[0]["conf"] == "0.9" and rows[1]["conf"] == "0.5"
    for row in rows:
        assert -1 <= row["x1"] < row["x2"] <= 321
        assert -1 <= row["y1"] < row["y2"] <= 241
    for crop, row in zip(crops, rows):
        assert crop["obj_id"] == row["obj_id"] and crop["frame"] == row["frame"]
    with tempfile.TemporaryDirectory() as tmp:
        npz = os.path.join(tmp, "c00.npz")
        write_masks_npz(npz, crops, (320, 240))
        back = load_mask(npz, 1)
        assert back.shape == (100, 100) and back.sum() == full.sum()
        assert load_mask(npz, 3).sum() == 0
    assert ground_point(full, (400, 400)) == (60.0, 236.0)
    rng = list(range(len([line0, line1])))
    assert len(rng) == 2
    import random
    shuffled = [line2, line1, line0]
    random.Random(0).shuffle(shuffled)
    rows2, _, _ = parse_events([blob(l) for l in shuffled], chunk, "person")
    assert [r["obj_id"] for r in rows2] == [r["obj_id"] for r in rows]
    try:
        parse_events([42], chunk, "person")
    except ParseError:
        pass
    else:
        raise AssertionError("non-dict event accepted")
    assert parse_events([{"type": "response.created"}], chunk, "person")[0] == []
    flat = ("<0f>3<|box;x1=5;y1=5;x2=5;y2=9;w=320;h=240|>"
            "<|mask;x=0;y=0;data=1,1,!00|>")
    rows_d, _, stats_d = parse_events([blob(flat)], chunk, "person")
    assert rows_d == [] and stats_d["dropped"] == 1
    for lineno, bad in [
        ("no header", "3<|box;x1=0;y1=0;x2=1;y2=1;w=320;h=240|>"
                      "<|mask;x=0;y=0;data=1,1,!00|>"),
        ("out of range", "<100f>3<|box;x1=0;y1=0;x2=1;y2=1;w=320;h=240|>"
                         "<|mask;x=0;y=0;data=1,1,!00|>"),
        ("bad record", "<0f>oops"),
        ("offset mask", "<0f>3<|box;x1=0;y1=0;x2=1;y2=1;w=320;h=240|>"
                        "<|mask;x=5;y=0;data=1,1,!00|>"),
        ("lossless marker", "<0f>3<|box;x1=0;y1=0;x2=1;y2=1;w=320;h=240|>"
                            "<|mask;x=0;y=0;data=1,1,~00|>"),
    ]:
        try:
            parse_events([blob(bad)], chunk, "person")
        except ParseError:
            pass
        else:
            raise AssertionError(f"{lineno} accepted")
    half = dict(chunk, width=160, height=120)
    rows_h, _, stats_h = parse_events([blob(line0)], half, "person")
    assert stats_h["scaled"] and rows_h[0]["x2"] <= 161
    import json as _json
    fix_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "fixtures", "sam")
    with open(os.path.join(fix_dir, "smoke-deltas.jsonl")) as fh:
        real_events = [_json.loads(line) for line in fh if line.strip()]
    real_chunk = {"chunk_id": "g1-20261104-c00", "session_id": "g1-20261104",
                  "t0_s": 0.0, "n_frames": 100, "width": 1920, "height": 1080}
    joined = [{"type": "response.output_text.delta",
               "delta": "\n".join(e["delta"] for e in real_events
                                   if e.get("type") == "response.output_text.delta")}]
    joined_rows, _, _ = parse_events(joined, real_chunk, "car")
    real_rows, real_crops, real_stats = parse_events(real_events, real_chunk, "car")
    assert [r["obj_id"] for r in joined_rows] == [r["obj_id"] for r in real_rows]
    assert len(real_rows) == 33, len(real_rows)
    assert real_stats["undecoded_masks"] == 33, real_stats  # alphabet TBD
    assert all(r["obj_id"].startswith("g1-20261104-c00:car:") for r in real_rows)
    assert all(r["conf"] != "" for r in real_rows)
    first = real_rows[0]
    assert (first["gx_px"], first["gy_px"]) == (1003.5, 146.0), (first["gx_px"], first["gy_px"])
    assert tuple(int(v) for v in real_crops[0]["mask_hw"]) == (15, 18)
    assert all(r["gx_px"] == (r["x1"] + r["x2"]) / 2 and r["gy_px"] == r["y2"]
               for r in real_rows)
    old_offline = os.environ.pop("SURI_OFFLINE", None)
    os.environ["SURI_CACHE_DIR"] = tempfile.mkdtemp(prefix="sam-cache-")
    os.environ["SURI_LEDGER"] = os.path.join(os.environ["SURI_CACHE_DIR"], "ledger.csv")
    hit_events = [
        blob(line0),
        {"type": "response.completed", "usage": {"video_frames_processed": 100}},
    ]
    est = estimate_usd(100)

    def send():
        yield from hit_events

    key_body = build_key_body("person", "abc123", 100)
    first = list(ledger.call("t-smoke", MODEL, ENDPOINT, key_body, send,
                            estimate_usd=est, cache_reads=False))
    second = list(ledger.call("t-smoke", MODEL, ENDPOINT, key_body, send,
                             estimate_usd=est, cache_reads=True))
    assert first == second and len(first) == 2
    import csv as _csv
    with open(os.environ["SURI_LEDGER"], newline="") as fh:
        ledger_rows = list(_csv.DictReader(fh))
    assert ledger_rows[-1]["cache_hit"] == "1" and float(ledger_rows[-1]["cost_usd"]) == 0
    assert list(ledger_rows[0].keys()) == list(ledger.LEDGER_COLUMNS)
    real_caps, ledger.CAPS = dict(ledger.CAPS), {"sam-3.1": 0.0}
    try:
        list(ledger.call("t-cap", MODEL, ENDPOINT, build_key_body("car", "zzz", 100),
                         send, estimate_usd=est, cache_reads=True))
    except ledger.CapExceeded:
        pass
    else:
        raise AssertionError("cap 0 accepted")
    finally:
        ledger.CAPS = real_caps
    try:
        list(ledger.call("t-contrib", "sam-3.1-contributor", ENDPOINT, key_body,
                         send, estimate_usd=est))
    except ValueError:
        pass
    else:
        raise AssertionError("contributor accepted")
    os.environ["SURI_OFFLINE"] = "1"
    try:
        list(ledger.call("t-off", MODEL, ENDPOINT, build_key_body("car", "qqq", 100),
                         send, estimate_usd=est, cache_reads=True))
    except ledger.CacheMiss:
        pass
    else:
        raise AssertionError("offline miss accepted")
    finally:
        if old_offline is None:
            os.environ.pop("SURI_OFFLINE", None)
        else:
            os.environ["SURI_OFFLINE"] = old_offline
    assert estimate_usd(10) == 50 * 0.0002
    try:
        check_concept("Person, car")
    except ValueError:
        pass
    else:
        raise AssertionError("bad concept accepted")
    chunks = [{"chunk_id": "g1-20261104-c01"},
              {"chunk_id": "g1-20261104-c02"},
              {"chunk_id": "g1-20261104-c03"}]
    ranged = [c["chunk_id"] for c in parse_selector("person:c01-03", chunks)[1]]
    assert ranged == ["g1-20261104-c01", "g1-20261104-c02", "g1-20261104-c03"], ranged
    literal = parse_selector("person:g1-20261104-c02", chunks)[1]
    assert literal == [chunks[1]], literal


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAM 3.1 client (spec 02).")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("selectors", nargs="*")
    parser.add_argument("--task", default=None)
    parser.add_argument("--window", default=None,
                        help="frame window F0-F1 cut from each chunk (smoke)")
    parser.add_argument("--draw", default=None,
                        help="write an overlay PNG of --frame to this path")
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        print("sam --check: PASS (grammar, ground point, masks, replay, cap)")
        return
    if not args.run_dir or not args.task:
        parser.error("need RUN_DIR and --task (or --check)")
    window = None
    if args.window:
        try:
            f0_s, _, f1_s = args.window.partition("-")
            window = (int(f0_s), int(f1_s))
            assert 0 <= window[0] < window[1]
        except (ValueError, AssertionError):
            parser.error("--window must be F0-F1 with 0 <= F0 < F1")
    try:
        code = run_batch(args.run_dir, args.selectors, args.task, window)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.draw:
        tracks_path = os.path.join(args.run_dir, "tracks.csv")
        with open(tracks_path, newline="") as fh:
            frame_rows = [r for r in csv.DictReader(fh)
                          if int(r["frame"]) == args.frame]
        if not frame_rows:
            print(f"draw: no rows on frame {args.frame}; skipping {args.draw}")
        else:
            chunk_ids = sorted({r["chunk_id"] for r in frame_rows})
            overlay_frame(chunk_mp4_path(args.run_dir, chunk_ids[0]),
                          args.frame, frame_rows, args.draw)
            print(f"drew {args.draw}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
