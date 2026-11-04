"""Wonder verification agent: verify candidates, write findings (spec 06).

verify scope: kinematics + segment/explain tools per candidate, verdicts
by the section 4.5 table, trace.jsonl. write scope: fixed numeral-free
templates, findings.json. Works offline: every network tool degrades to
its labeled fallback (respaldo / no es en vivo), and the table decides
from flags alone when explain is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import ledger

WONDER_MODEL = "muse-spark-1.3"
WONDER_ENDPOINT = "meta/responses"
WONDER_BASE_URL = "https://api.meta.ai/v1"
EXPLAIN_MODEL = "meta-models/muse-spark-open"
EXPLAIN_ENDPOINT = "meta/explain"
BASETEN_API_MODEL = "meta-models/muse-spark-open-1026"
BASETEN_LEDGER_MODEL = "baseten/muse-spark-open"

CROSSING_ID = "mty-garciaroel-ahuehuetes"

FLAG_CODES = ("id_switch", "mask_jump", "impossible_speed", "occlusion",
              "paths_do_not_cross", "parked_or_static", "outside_phase_logic")
REJECT_CODES = FLAG_CODES + ("explain_disagrees",)
HARD_FLAGS = ("id_switch", "mask_jump", "impossible_speed",
              "paths_do_not_cross", "parked_or_static")
SOFT_FLAGS = ("occlusion", "outside_phase_logic")

EXPLAIN_METRIC_KEYS = ("pet_s", "tier", "phase_at_conflict",
                       "first.speed_mps", "second.speed_mps",
                       "second.movement")
EXPLAIN_CONFLICT_TYPES = ("right_turn", "left_turn", "through", "other")
EXPLAIN_CONFIDENCES = ("low", "medium", "high")
CONCEPT_REFUSED = ("face", "plate", "cloth", "dress", "shirt")

REFERENCES = [{"ref_id": "fhwa-hrt-18-044",
               "citation": "FHWA-HRT-18-044 (leading pedestrian interval "
                           "evaluation)",
               "url": None,
               "supports": ["lpi"]}]
GUIDELINE_REFS = {"lpi": ["fhwa-hrt-18-044"]}

PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*)\}\}")
FINDING_RE = re.compile(r"F-\d+")
EVENT_RE = re.compile(r"E-\d+")

NUMBER_WORDS = {
    "es": ("cero", "dos", "tres", "cuatro", "cinco", "seis", "siete",
           "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
           "quince", "dieciseis", "dieciséis", "diecisiete", "dieciocho",
           "diecinueve", "veinte", "treinta", "cuarenta", "cincuenta",
           "sesenta", "setenta", "ochenta", "noventa", "cien", "ciento",
           "cientos", "mil", "millon", "millón", "millones", "por ciento"),
    "en": ("zero", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
           "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
           "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
           "eighty", "ninety", "hundred", "thousand", "million",
           "percent", "dozen"),
}

BANNED = {
    "es": (r"reducir[áa](?:n)?", r"disminuir[áa](?:n)?",
           r"evitar[áa](?:n)?", r"prevenir[áa](?:n)?",
           r"salvar[áa](?:n)?", r"garantiza", r"predic\w*",
           r"tiempo real", r"velocidades validadas", r"cualquier video",
           r"el estado m[áa]s peligroso", r"avalad\w*"),
    "en": (r"will reduce", r"will prevent", r"will save", r"predict\w*",
           r"real[- ]time", r"validated speeds", r"any video",
           r"most dangerous state", r"guarantee", r"endorse\w*"),
}


def lookup(ctx, path):
    """Resolve a placeholder path against the render context."""
    node = ctx
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(path)
    return node


def fmt_value(name, value):
    """Spec 06 section 4.8: null as em dash, integers plain, floats by
    name. A _lo/_hi key prints like its base key."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        last = name.split(".")[-1]
        for suffix in ("_lo", "_hi"):
            if last.endswith(suffix) and len(last) > len(suffix):
                return fmt_value(name[: -len(suffix)], value)
        if last.endswith(("_s", "_s_sens", "_m", "_mps", "_per_h", "_pct",
                          "_pts", "hours", "hours_observed")):
            return f"{value:.1f}"
        if last.startswith(("share_", "precision", "recall", "f1", "pa")):
            return f"{value * 100:.0f}%"
        return f"{value:.2f}"
    return str(value)


def render(text, ctx):
    """Replace every placeholder with its formatted context value."""
    def sub(match):
        return fmt_value(match.group(1), lookup(ctx, match.group(1)))
    return PLACEHOLDER_RE.sub(sub, text)


def validate_text(field, text, lang, ctx):
    """V1-V4 on one prose field. Returns a list of error strings."""
    errors = []
    for match in PLACEHOLDER_RE.finditer(text):
        path = match.group(1)
        if path.startswith("metrics."):
            continue
        try:
            lookup(ctx, path)
        except KeyError:
            errors.append(f"{field}: unresolved {{{{{path}}}}}")
    bare = PLACEHOLDER_RE.sub("", text)
    if re.search(r"\d", bare):
        errors.append(f"{field}: digit")
    words = NUMBER_WORDS.get(lang, ())
    lowered = f" {bare.lower()} "
    for word in words:
        if re.search(rf"(?<![a-záéíóúñ]){re.escape(word)}(?![a-záéíóúñ])", lowered):
            errors.append(f"{field}: number_word {word}")
            break
    for pattern in BANNED.get(lang, ()):
        if re.search(pattern, bare, re.IGNORECASE):
            errors.append(f"{field}: claim {pattern}")
            break
    return errors


LPI_SET = (0, 3, 5)
PED_INSERT_S = 1.5
CLEAR_PET_S = 3.0


def counterfactual(events, lpi_s):
    """Mechanical LPI replay: effective PET = pet + insert + lpi.

    Cleared when the effective PET reaches 3.0 s. Counts run over
    pre-verification candidates.
    """
    serious = close = cleared = 0
    for event in events:
        effective = event["pet_s"] + PED_INSERT_S + lpi_s
        if effective >= CLEAR_PET_S:
            cleared += 1
            continue
        if event["tier"] == "serious":
            serious += 1
        else:
            close += 1
    return {"lpi_s": lpi_s, "serious": serious, "close": close,
            "cleared": cleared, "n": len(events)}


def pick_lpi(results):
    """argmin over (serious, close, lpi_s); 0 unless it beats baseline."""
    base = results["baseline"]
    key = lambda r: (r["serious"], r["close"], r["lpi_s"])
    best = min((results[k] for k in sorted(results) if k != "baseline"),
               key=key)
    if key(best)[:2] >= key(base)[:2]:
        return 0
    return best["lpi_s"]


def counterfactual_doc(events):
    """counterfactual.json grounds the LPI recommendation (cacheable)."""
    res = {k: counterfactual(events, k) for k in LPI_SET}
    results = {"baseline": res[0], "lpi_3": res[3], "lpi_5": res[5]}
    pick = pick_lpi(results)
    ref = "counterfactual.baseline" if pick == 0 else f"counterfactual.lpi_{pick}"
    return {"scenario": "escenario mecánico",
            "ped_insert_s": PED_INSERT_S, "clear_pet_s": CLEAR_PET_S,
            "results": results, "pick_lpi_s": pick, "pick_ref": ref,
            "clearance": []}


def kinematics_tool(event_id, session):
    """Candidate fields for one event id. Pure file read."""
    for event in session["events"]:
        if event["event_id"] == event_id:
            return dict(event)
    raise ValueError(f"unknown event {event_id}")


def check_concept(concept):
    """Spec 06 section 4.3: sam grammar plus no faces, plates or clothing."""
    import sam

    sam.check_concept(concept)
    lowered = concept.lower()
    if any(word in lowered for word in CONCEPT_REFUSED):
        raise ValueError(f"refused concept {concept!r}")
    return concept


def segment_tool(clip_id, concept, session, task_id, cache_reads=True):
    """Rematch the event window with an agent-written concept.

    Cuts the window from the unblurred analysis chunk (measurement path,
    same as sam), sends one concept request through the ledger, and
    writes the tracks CSV. The rematch stays null without a calibrated
    projection: it informs the verdict and never rewrites events.json.
    """
    import sam

    check_concept(concept)
    event = kinematics_tool(clip_id, session)
    chunk = session["chunks"][event["chunk_id"]]
    a, b = (float(v) for v in event["window_s"])
    t0 = float(chunk["t0_s"])
    f0 = max(0, int(round((a - t0) * sam.FPS)))
    f1 = max(f0 + 1, int(round((b - t0) * sam.FPS)))
    n_frames = f1 - f0
    sha = event["chunk_sha"] if "chunk_sha" in event else chunk["sha256"]
    key_body = sam.build_key_body(
        concept, f"{sha}:{f0}-{f1}:{sam.VARIANT}", n_frames)
    est = sam.estimate_usd(n_frames)
    dst = os.path.join(session["tmp"], f"{clip_id}.mp4")

    def send():
        os.makedirs(session["tmp"], exist_ok=True)
        client = sam.get_client()
        media = sam.cut_window(
            sam.chunk_mp4_path(session["run_dir"], event["chunk_id"]),
            f0, f1, dst)
        fid = sam.upload_chunk(client, media)
        return sam.send_request(client, fid, concept)

    events = list(ledger.call(task_id, sam.MODEL, sam.ENDPOINT, key_body,
                              send, estimate_usd=est, cache_reads=cache_reads))
    chunk_info = dict(chunk, chunk_id=clip_id, session_id=event["session_id"],
                      t0_s=a, n_frames=n_frames,
                      width=int(chunk["width"]), height=int(chunk["height"]))
    rows, _crops, stats = sam.parse_events(events, chunk_info, concept)
    slug = concept.replace(" ", "_")
    tracks_ref = f"segment/{clip_id}-{slug}.csv"
    seg_dir = os.path.join(session["run_dir"], "segment")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(session["run_dir"], tracks_ref), "w",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sam.TRACKS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    by_obj = {}
    for row in rows:
        by_obj.setdefault(row["obj_id"], []).append(float(row["t_s"]))
    tracks = [{"obj_id": obj, "t_first_s": min(ts), "t_last_s": max(ts),
               "speed_mps": None, "in_conflict_area": None,
               "iou_first": None, "iou_second": None}
              for obj, ts in sorted(by_obj.items())]
    return {"clip_id": clip_id, "concept": concept, "frames": n_frames,
            "peak_simultaneous": stats["peak_ids"], "tracks": tracks,
            "rematch": None, "tracks_ref": tracks_ref,
            "masks_visible": len(by_obj),
            "empty_masks": stats["empty_masks"],
            "dropped": stats["dropped"]}


def _blurred_clip(session, clip_id):
    """Path of the publishable clip; enforces the blurred-v1 tag."""
    import subprocess

    crossing = os.path.dirname(session["run_dir"].rstrip(os.sep))
    path = os.path.join(crossing, "clips", f"{clip_id}.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no blurred clip {path}")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format_tags=comment", "-of", "default=noprint_wrappers=1:nokey=1",
         path], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or proc.stdout.strip() != "blurred-v1":
        raise ValueError(f"clip tag is not blurred-v1: {path}")
    return path


def _metric_value(event, key):
    """One of the six explain metric keys, dotted into the event."""
    node = event
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def explain_metrics(event):
    """The six metric keys explain receives, from events.json values."""
    return {key: _metric_value(event, key) for key in EXPLAIN_METRIC_KEYS}


def _clean_string(value, limit):
    """Null a tool string with digits or over the length limit."""
    if not isinstance(value, str) or len(value) > limit:
        return None, True
    if re.search(r"\d", value):
        return None, True
    return value, False


def _file_sha(path):
    """sha256 of a file, used as the explain cache media key."""
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_tool_calls(raw):
    """Join chat-completions tool calls from streamed deltas or one message."""
    complete = []
    partial = {}
    for event in raw:
        choice = (event.get("choices") or [{}])[0]
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            complete.append(call)
        for call in (choice.get("delta") or {}).get("tool_calls") or []:
            idx = call.get("index", 0)
            slot = partial.setdefault(
                idx, {"id": None, "function": {"name": "", "arguments": ""}})
            if call.get("id"):
                slot["id"] = call["id"]
            fn = call.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
    if partial:
        return [partial[i] for i in sorted(partial)]
    return complete


def explain_tool(clip_id, metrics, session, task_id, cache_reads=True):
    """Ask Spark what the blurred clip shows. Spec 06 section 4.3 shape.

    Sends the blurred clip (tag enforced) plus the six metric keys. The
    primary is Baseten Spark Open, retried once with a reminder; then
    muse-spark-1.3 at the Meta API (label respaldo). Both down gives
    explain_unavailable and verdicts use the flags only.
    """
    if os.environ.get("SURI_OFFLINE") == "1":
        return {"clip_id": clip_id, "model": None, "metrics_sent": metrics,
                "error": "explain_unavailable", "fallback": "no es en vivo",
                "detail": "offline"}
    clip_path = _blurred_clip(session, clip_id)
    media_sha = _file_sha(clip_path)
    prompt = ("You watch a short blurred clip from a fixed camera above a "
              "signalized crossing. Boxes mark the two road users of a "
              "candidate conflict. Code measured these metrics: "
              + json.dumps(metrics, sort_keys=True) + ". Decide whether this "
              "is a real pedestrian-vehicle conflict, where one passed "
              "through the spot the other had just left, or a false "
              "positive: a tracking error, a parked or waiting road user, "
              "or paths that never meet. Call report_conflict once. Write "
              "actors as short English noun phrases and likely_cause as one "
              "short Spanish phrase. No digits. Describe road users by type "
              "and visible aids only: no faces, clothing or plates.")
    tool_spec = {
        "type": "function",
        "function": {
            "name": "report_conflict",
            "description": "Report whether the clip shows a real conflict",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "conflict": {"type": "boolean"},
                    "conflict_type": {"type": "string"},
                    "actors": {"type": "array",
                               "items": {"type": "string"}},
                    "likely_cause": {"type": "string"},
                    "confidence": {"type": "string",
                                   "enum": ["low", "medium", "high"]},
                },
                "required": ["conflict", "confidence"],
            },
        },
    }
    attempts = [
        (BASETEN_API_MODEL, BASETEN_LEDGER_MODEL,
         os.environ.get("BASETEN_BASE_URL", "https://api.baseten.co/v1"),
         os.environ.get("BASETEN_API_KEY"), False, None),
        (BASETEN_API_MODEL, BASETEN_LEDGER_MODEL,
         os.environ.get("BASETEN_BASE_URL", "https://api.baseten.co/v1"),
         os.environ.get("BASETEN_API_KEY"), True, None),
        (WONDER_MODEL, WONDER_MODEL, WONDER_BASE_URL,
         os.environ.get("MODEL_API_KEY"), False, "respaldo"),
    ]
    last_error = "no tool call"
    for api_model, ledger_model, base_url, api_key, reminder, fallback in attempts:
        if not api_key:
            last_error = "missing key"
            continue
        endpoint = "baseten/explain" if fallback is None else EXPLAIN_ENDPOINT
        key_body = {"clip_id": clip_id, "metrics": metrics,
                    "media": media_sha, "reminder": reminder}

        def send(api_model=api_model, base_url=base_url, api_key=api_key,
                 reminder=reminder):
            import openai

            client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                   max_retries=0)
            with open(clip_path, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="user_data")
            file_id = getattr(uploaded, "id", None)
            if not file_id:
                raise ValueError(f"upload of {clip_path} returned no file id")
            text = prompt + (" Remember: call report_conflict once." if reminder
                             else "")
            body = {"model": api_model,
                    "messages": [{"role": "user",
                                  "content": [{"type": "text", "text": text},
                                              {"type": "input_video",
                                               "input_video":
                                                   {"file_id": file_id}}]}],
                    "tools": [tool_spec],
                    "tool_choice": {"type": "function",
                                    "function": {"name": "report_conflict"}}}
            stream = client.chat.completions.create(stream=True, **body)
            return [event.model_dump() for event in stream]

        try:
            raw = list(ledger.call(task_id, ledger_model, endpoint, key_body,
                                   send, estimate_usd=0.03,
                                   cache_reads=cache_reads))
        except Exception as exc:
            last_error = type(exc).__name__
            continue
        calls = _stream_tool_calls(raw)
        if not calls:
            last_error = "no tool call"
            continue
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except (ValueError, KeyError, TypeError):
            last_error = "bad arguments"
            continue
        if not isinstance(args.get("conflict"), bool):
            last_error = "bad conflict value"
            continue
        dropped = []
        ctype, bad = _clean_string(args.get("conflict_type"), 60)
        if bad or ctype not in EXPLAIN_CONFLICT_TYPES:
            ctype, dropped = None, dropped + ["conflict_type"]
        actors = args.get("actors")
        if (not isinstance(actors, list) or len(actors) > 4
                or any(not isinstance(a, str) or len(a) > 60
                       or re.search(r"\d", a) for a in actors)):
            actors, dropped = None, dropped + ["actors"]
        cause, bad = _clean_string(args.get("likely_cause"), 160)
        if bad:
            cause, dropped = None, dropped + ["likely_cause"]
        confidence = args.get("confidence")
        if confidence not in EXPLAIN_CONFIDENCES:
            confidence, dropped = None, dropped + ["confidence"]
        return {"clip_id": clip_id, "model": ledger_model,
                "metrics_sent": metrics,
                "conflict": args["conflict"],
                "false_positive": not args["conflict"],
                "conflict_type": ctype, "actors": actors,
                "likely_cause": cause, "confidence": confidence,
                "dropped": dropped, "fallback": fallback}
    return {"clip_id": clip_id, "model": None, "metrics_sent": metrics,
            "error": "explain_unavailable", "fallback": None,
            "detail": last_error}


def plausibility(event, rows_by_id):
    """Flags for one event, computed by kinematics (spec 06 section 4.4).

    rows_by_id maps obj_id to load_tracks-style row dicts. Kept as a
    thin adapter so the verify loop does not depend on track internals.
    """
    import kinematics

    tracks = [{"obj_id": obj_id, "cls": "", "session_id": "",
               "rows": [{"T": int(round(float(r["t_s"]) * 10))} for r in rows]}
              for obj_id, rows in rows_by_id.items()]
    return kinematics.plausibility(event, tracks)


def counterfactual_tool(lpi_s, crossing_dir):
    """Read counterfactual.json: baseline and the lpi_s entry (spec 4.3)."""
    if lpi_s not in (3, 5, 7):
        return {"error": "scope", "detail": f"bad lpi_s {lpi_s}"}
    with open(os.path.join(crossing_dir, "counterfactual.json")) as fh:
        doc = json.load(fh)
    entry = doc["results"].get(f"lpi_{lpi_s}")
    if entry is None:
        return {"error": "scope", "detail": f"no lpi_{lpi_s} entry"}
    return {"scenario": doc["scenario"], "lpi_s": lpi_s,
            "baseline": doc["results"]["baseline"], "result": entry,
            "corrections": []}


def decide(flags, explain, segment_clean=False):
    """Section 4.5 table. Returns (status, [reason_codes])."""
    hard = sorted(c for c in HARD_FLAGS if flags.get(c) is not None)
    if hard:
        return "rejected", hard
    soft = sorted(c for c in SOFT_FLAGS if flags.get(c) is not None)
    fp = bool(explain and explain.get("false_positive") is True)
    high = bool(explain and explain.get("confidence") == "high")
    if fp:
        if not soft:
            if high and segment_clean:
                return "rejected", ["explain_disagrees"]
            return "verified", []
        return "rejected", soft
    return "verified", []


def check_submit(scope_ids, verdicts, info):
    """C1-C4. info maps event_id to flags/explain/segment evidence.

    Returns a list of error strings; empty means the batch may commit.
    """
    errors = []
    seen = [v["event_id"] for v in verdicts]
    if sorted(seen) != sorted(scope_ids) or len(set(seen)) != len(seen):
        errors.append("C1: verdicts do not match the candidate scope")
    for verdict in verdicts:
        eid = verdict["event_id"]
        status, codes = verdict["status"], verdict["reason_codes"]
        if status not in ("verified", "rejected"):
            errors.append(f"C1: {eid} bad status {status}")
            continue
        have = info.get(eid)
        if have is None or have.get("kinematics") is not True:
            errors.append(f"C2: {eid} no kinematics step")
            continue
        flags, explain = have["flags"], have["explain"]
        segment_clean = have.get("segment_clean", False)
        if status == "verified":
            if decide(flags, explain, segment_clean) != ("verified", []):
                errors.append(f"C3: {eid} verified against the table")
            if explain is None and have.get("explain_error") is not True:
                errors.append(f"C3: {eid} verified with no explain call")
        else:
            if not codes or any(c not in REJECT_CODES for c in codes):
                errors.append(f"C4: {eid} bad codes {codes}")
                continue
            hard = sorted(c for c in HARD_FLAGS if flags.get(c) is not None)
            soft = sorted(c for c in SOFT_FLAGS if flags.get(c) is not None)
            fp = bool(explain and explain.get("false_positive") is True)
            high = bool(explain and explain.get("confidence") == "high")
            allowed = (codes == hard and bool(hard)) or \
                (codes == soft and bool(soft) and fp) or \
                (codes == ["explain_disagrees"] and fp and high
                 and segment_clean)
            if not allowed:
                errors.append(f"C4: {eid} codes unsupported {codes}")
    return errors


def cluster_events(events):
    """Group verified events into encounters by window overlap.

    Single-phase pixel runs have no movement or phase keys, so clusters
    are connected components of overlapping clip windows.
    """
    order = sorted(range(len(events)),
                   key=lambda i: float(events[i]["window_s"][0]))
    parent = list(range(len(events)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            ai, bi = (float(v) for v in events[i]["window_s"])
            aj, bj = (float(v) for v in events[j]["window_s"])
            if aj <= bi and ai <= bj:
                union(i, j)
    groups = {}
    for i, event in enumerate(events):
        groups.setdefault(find(i), []).append(event)
    return [sorted(g, key=lambda e: e["event_id"]) for _, g in
            sorted(groups.items(), key=lambda kv: kv[0])]


def fill_clusters(clusters, outcomes):
    """Count explain conflict types and causes per cluster (max 5)."""
    for cluster in clusters:
        types, causes = {}, []
        for eid in cluster["event_ids"]:
            out = outcomes.get(eid, {})
            ctype = out.get("conflict_type") or ""
            if ctype:
                types[ctype] = types.get(ctype, 0) + 1
            cause = out.get("likely_cause") or ""
            if cause and cause not in causes and len(causes) < 5:
                causes.append(cause)
        cluster["conflict_types"] = types
        cluster["causes"] = causes
    return clusters


def build_clusters(verified, hours_observed=0.0):
    """Cluster records for findings.

    Encounters are connected components of overlapping clip windows:
    with pixel-only inputs every movement and phase key is null, so the
    section 4.6 key would merge the whole session into one cluster. The
    clusters carry the section 4.6 fields and sort by serious, size, key.
    """
    clusters = []
    for group in cluster_events(verified):
        pets = sorted(e["pet_s"] for e in group)
        n_serious = sum(1 for e in group if e["tier"] == "serious")
        med = pets[len(pets) // 2]
        second = group[0]["second"]
        first = group[0]["first"]
        car = second if second["cls"] == "car" else first
        clusters.append({
            "event_ids": [e["event_id"] for e in group],
            "conflict_area": group[0]["conflict_area"],
            "crosswalk": group[0].get("crosswalk"),
            "movement": car.get("movement"),
            "phase": group[0].get("phase_at_conflict"),
            "n_events": len(group), "n_serious": n_serious,
            "n_close": len(group) - n_serious,
            "pet_min_s": pets[0], "pet_median_s": med,
            "serious_per_h": round(n_serious / hours_observed, 1)
            if hours_observed else 0.0,
            "conflict_types": {}, "causes": [],
            "example_event_id":
                min(group, key=lambda e: e["pet_s"])["event_id"],
            "masks": sorted({f"{e['session_id']}/masks/"
                             f"{e['chunk_id']}.npz" for e in group}),
        })
    key = lambda c: (-c["n_serious"], -c["n_events"],
                     (c["conflict_area"], c["movement"] or "",
                      c["phase"] or ""))
    clusters.sort(key=key)
    for i, cluster in enumerate(clusters, 1):
        cluster["cluster_id"] = f"F-{i:02d}"
    return clusters


FINDING_TEMPLATE = ("Zona de conflicto {{F.conflict_area}}. Encuentros "
                    "verificados: {{F.n_events}}. Serios: {{F.n_serious}}. "
                    "PET mínimo: {{F.pet_min_s}} s.")
ES_TEMPLATE = ("Se verificaron candidatos con conflicto entre peatón y "
               "vehículo. La causa más frecuente fue {{run.top_cause}}. "
               "La decisión queda en manos del equipo de ingeniería.")
EN_TEMPLATE = ("Verified candidates showed pedestrian-vehicle conflict. "
               "The most common cause was {{run.top_cause}}. The decision "
               "rests with the engineering team.")


def build_findings(clusters, cf_doc, run_ctx, sessions):
    """findings.json from clusters. Fixed templates, placeholders only."""
    pick = cf_doc["pick_lpi_s"]
    pick_key = "baseline" if pick == 0 else f"lpi_{pick}"
    rec_ids = ["R-01"] if pick > 0 else []
    findings = []
    for c in clusters:
        finding = {"finding_id": c["cluster_id"],
                   "cluster_id": c["cluster_id"],
                   "event_ids": c["event_ids"],
                   "text_template": FINDING_TEMPLATE,
                   "evidence": {
                       "clips": [f"clips/{eid}.mp4" for eid in c["event_ids"]],
                       "keyframes": [f"clips/{eid}.jpg"
                                     for eid in c["event_ids"]]}}
        if rec_ids:
            refs = [f"{c['cluster_id']}.n_serious",
                    f"{c['cluster_id']}.pet_min_s"] + \
                [f"{eid}.pet_s" for eid in c["event_ids"]]
            finding["evidence"]["masks"] = c.get("masks", [])
            finding["evidence"]["metric_refs"] = refs
            finding["recommendation_ids"] = rec_ids
            finding["references"] = list(GUIDELINE_REFS["lpi"])
        findings.append(finding)
    recommendations = []
    if pick > 0:
        recommendations.append({
            "recommendation_id": "R-01", "kind": "lpi",
            "finding_ids": [c["cluster_id"] for c in clusters],
            "counterfactual_ref": f"counterfactual.{pick_key}",
            "guideline_references": list(GUIDELINE_REFS["lpi"]),
            "text_template":
                ("Se recomienda evaluar un intervalo peatonal adelantado "
                 f"(LPI) de {{{{counterfactual.{pick_key}.lpi_s}}}} s para "
                 "los cruces peatonales. En el escenario mecánico, los "
                 "conflictos serios candidatos pasarían de "
                 "{{counterfactual.baseline.serious}} a "
                 f"{{{{counterfactual.{pick_key}.serious}}}}."),
            "references": list(GUIDELINE_REFS["lpi"])})
    es_tail = " La decisión queda en manos del equipo de ingeniería."
    en_tail = " The decision rests with the engineering team."
    summary_es, summary_en = ES_TEMPLATE[:-len(es_tail)], EN_TEMPLATE[:-len(en_tail)]
    if recommendations:
        summary_es += " Se propone como medida un intervalo peatonal adelantado."
        summary_en += " The proposed measure is a leading pedestrian interval."
    elif cf_doc.get("signalized") is False:
        summary_es += (" El cruce de la demostración no tiene semáforo, así "
                       "que no se evalúa un intervalo peatonal adelantado.")
        summary_en += (" The demo crossing has no traffic signal, so a "
                       "leading pedestrian interval is not evaluated.")
    summary_es, summary_en = summary_es + es_tail, summary_en + en_tail
    if not recommendations and cf_doc.get("signalized") is not False:
        summary_es += (" En el escenario mecánico, los conflictos serios "
                       "candidatos serían {{counterfactual.baseline.serious}}.")
        summary_en += (" In the mechanical scenario, the candidate serious "
                       "conflicts would be {{counterfactual.baseline.serious}}.")
    run_ctx = dict(run_ctx, n_recommendations=len(recommendations))
    return {"crossing_id": run_ctx.get("crossing_id") or CROSSING_ID, "sessions": sessions,
            "run": run_ctx, "summary": {"es": summary_es, "en": summary_en},
            "findings": findings, "recommendations": recommendations,
            "references": REFERENCES}


def _resolve_ref(ctx, ref):
    """Resolve a dotted counterfactual_ref inside ctx."""
    if not ref.startswith("counterfactual."):
        raise KeyError(ref)
    node = ctx["counterfactual"]
    for part in ref.split(".")[1:]:
        node = node[part]
    return node


def validate(doc, ctx):
    """V1-V6 over findings.json. Returns a list of error strings."""
    errors = []
    clusters = ctx.get("clusters", {})
    statuses = ctx.get("statuses", {})
    crossing = ctx.get("crossing_dir", "")
    cfc = ctx.get("counterfactual", {})
    run = ctx.get("run", {})
    findings = doc.get("findings", [])
    for finding in findings:
        fid = finding["finding_id"]
        cluster = clusters.get(finding.get("cluster_id", ""), {})
        fctx = dict(ctx, F=cluster)
        template = finding["text_template"]
        if len(template) > 900:
            errors.append(f"{fid}: over 900 chars")
        errors += validate_text(fid, template, "es", fctx)
        for cited in FINDING_RE.findall(template):
            if cited != finding["cluster_id"]:
                errors.append(f"{fid}: cites {cited} outside its cluster")
        for cited in EVENT_RE.findall(template):
            if cited not in cluster.get("event_ids", []):
                errors.append(f"{fid}: cites {cited} outside its cluster")
        if sorted(finding.get("event_ids", [])) != \
                sorted(cluster.get("event_ids", [])):
            errors.append(f"{fid}: events differ from its cluster")
        evidence = finding.get("evidence", {})
        for eid in finding.get("event_ids", []):
            if statuses.get(eid) != "verified":
                errors.append(f"{fid}: cites unverified {eid}")
        for clip in evidence.get("clips", []):
            if not os.path.exists(os.path.join(crossing, clip)):
                errors.append(f"{fid}: missing clip {clip}")
    for rec in doc.get("recommendations", []):
        rid = rec["recommendation_id"]
        if len(rec["text_template"]) > 900:
            errors.append(f"{rid}: over 900 chars")
        errors += validate_text(rid, rec["text_template"], "es", ctx)
        for fid in rec.get("finding_ids", []):
            if fid not in [f["finding_id"] for f in findings]:
                errors.append(f"{rid}: cites unknown {fid}")
        ref = rec.get("counterfactual_ref", "")
        try:
            _resolve_ref(ctx, ref)
        except KeyError:
            errors.append(f"{rid}: bad counterfactual_ref {ref}")
        if rec.get("kind") == "lpi":
            pick = cfc.get("pick_lpi_s")
            want = ("counterfactual.baseline" if pick == 0
                    else f"counterfactual.lpi_{pick}")
            if ref != want:
                errors.append(f"{rid}: ref {ref} is not the pick {want}")
        refs = rec.get("guideline_references", [])
        if not refs or refs[0] not in GUIDELINE_REFS.get(rec.get("kind"),
                                                         []):
            errors.append(f"{rid}: guideline does not support its kind")
    have = {f["cluster_id"] for f in findings}
    if set(clusters) != have:
        errors.append("clusters without exactly one finding: "
                      f"{sorted(set(clusters) ^ have)}")
    summary = doc.get("summary", {})
    for lang, limit in (("es", 1200), ("en", 900)):
        text = summary.get(lang, "")
        rendered = render(text, ctx)
        if len(rendered) > limit:
            errors.append(f"summary.{lang}: over {limit} chars")
        errors += validate_text(f"summary.{lang}", text, lang, ctx)
    return errors


TRACE_FIELDS = ("step_id", "ts", "model", "tool", "args", "result_ref",
                "latency_ms", "tokens_in", "tokens_out", "frames",
                "cost_usd", "cache_hit", "fallback")


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _chunk_end(chunk):
    if chunk.get("t1_s") not in (None, ""):
        return float(chunk["t1_s"])
    return float(chunk["t0_s"]) + float(chunk.get("duration_s") or 0.0)


def _union_hours(chunks):
    """Observed hours counting chunk overlaps once (spec 06 section 4.6)."""
    spans = sorted((float(c["t0_s"]), _chunk_end(c)) for c in chunks)
    total, cur = 0.0, None
    for start, end in spans:
        if cur is None or start >= cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [start, end]
        elif end > cur[1]:
            cur[1] = end
    if cur is not None:
        total += cur[1] - cur[0]
    return round(total / 3600.0, 3)


def load_session(run_dir):
    """Read events, tracks, chunks, zones for one session run."""
    with open(os.path.join(run_dir, "events.json")) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        events, wrapped = raw, False
    else:
        events, wrapped = raw["events"], True
    rows_by_id = {}
    with open(os.path.join(run_dir, "tracks.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            rows_by_id.setdefault(row["obj_id"], []).append(row)
    chunks = {}
    with open(os.path.join(run_dir, "chunks.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            chunks[row["chunk_id"]] = row
    with open(os.path.join(run_dir, "zones.json")) as fh:
        zones = json.load(fh)
    crosswalks = {ca["id"]: ca.get("crosswalk")
                   for ca in zones.get("conflict_areas", [])}
    return {"run_dir": run_dir,
            "session_id": os.path.basename(run_dir.rstrip(os.sep)),
            "events": events, "events_wrapped": wrapped,
            "rows_by_id": rows_by_id, "chunks": chunks,
            "crosswalks": crosswalks,
            "tmp": os.path.join(run_dir, "tmp")}


def _ledger_rows():
    path = os.environ.get("SURI_LEDGER", os.path.join("runs", "ledger.csv"))
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class _Scope:
    """One budgeted conversation: steps, result files, spend attribution.

    conv_id is the chunk_id for verify, "write" for the write pass.
    Every tool result lands in steps/<conv_id>/sNN.json, the step's
    result_ref. Over budget the tool returns budget_exhausted.
    """

    def __init__(self, run_dir, conv_id, task_id, budget, replay):
        self.run_dir = run_dir
        self.conv_id = conv_id
        self.task_id = task_id
        self.budget = budget
        self.replay = replay
        self.calls = 0
        self.order = 0
        self.steps = []

    def _spend(self, n0):
        new_rows = _ledger_rows()[n0:]
        cost = round(sum(float(r.get("cost_usd") or 0) for r in new_rows), 6)
        hits = [int(r.get("cache_hit") or 0) for r in new_rows]
        if not new_rows:
            return cost, None
        return cost, bool(all(hits))

    def step(self, tool, model, args, fn, frames=None):
        """Run fn, record the spec 06 step, return the tool result."""
        self.order += 1
        latency_ms, cost, hit = 0, 0.0, None
        if self.calls >= self.budget:
            result = {"error": "budget_exhausted"}
        else:
            self.calls += 1
            n0 = len(_ledger_rows())
            tick = time.time()
            try:
                result = fn()
            except Exception as exc:
                result = {"error": _tool_error(tool, exc)}
            latency_ms = int(round((time.time() - tick) * 1000))
            cost, hit = self._spend(n0)
        if isinstance(result, dict) and result.get("fallback"):
            fallback = result["fallback"]
        else:
            fallback = "no es en vivo" if self.replay else None
        result_ref = f"steps/{self.conv_id}/s{self.order:02d}.json"
        path = os.path.join(self.run_dir, result_ref)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        self.steps.append(
            {"step_id": f"{self.conv_id}/s{self.order:02d}",
             "ts": _utcnow(), "model": model, "tool": tool, "args": args,
             "result_ref": result_ref, "latency_ms": latency_ms,
             "tokens_in": None, "tokens_out": None, "frames": frames,
             "cost_usd": cost, "cache_hit": hit, "fallback": fallback})
        return result


def _tool_error(tool, exc):
    """Map an exception to the spec 06 tool error vocabulary."""
    name = type(exc).__name__
    if tool == "segment" and "Cache" in name:
        return "sam_unavailable"
    return name


def _clip_frames(event, chunk):
    """Explain clip frames from the window span and the chunk fps."""
    a, b = (float(v) for v in event["window_s"])
    fps = float(chunk.get("fps") or 10)
    return max(1, int(round((b - a) * fps)))


def _event_names(step):
    """Event ids a step's args name (for verify.tool_calls)."""
    args = step.get("args", {})
    return {args.get("event_id"), args.get("clip_id")} - {None}


def run_verify(run_dir, task_id, replay=False, force=()):
    """Verify every candidate: steps, table verdicts, trace + events.

    One conversation per chunk (Turn 1: kinematics and explain for
    every candidate; then segment where the table may need it), closed
    by a submit step that re-checks C1-C4. Budget breaches and
    exceptions leave the candidate for n_unverified and fail loudly
    only when the submit checks do.
    """
    session = load_session(run_dir)
    replay = replay or os.environ.get("SURI_OFFLINE") == "1"
    verdicts, info = [], {}
    steps, concepts = [], set()
    counts = {"n_candidates": 0, "n_verified": 0, "n_rejected": 0,
              "n_unverified": 0, "cost_usd": 0.0}
    by_reason = {}
    groups = {}
    for event in session["events"]:
        if event["status"] == "candidate":
            groups.setdefault(event["chunk_id"], []).append(event)
    for chunk_id in sorted(groups):
        group = groups[chunk_id]
        scope = _Scope(run_dir, chunk_id, task_id,
                       2 * len(group) + 8, replay)
        kins = {}
        for event in group:
            eid = event["event_id"]
            kin = scope.step("kinematics", None, {"event_id": eid},
                             lambda e=eid: kinematics_tool(e, session))
            if isinstance(kin, dict) and kin.get("error"):
                counts["n_unverified"] += 1
                continue
            try:
                kins[eid] = (kin, plausibility(kin, session["rows_by_id"]))
            except Exception:
                counts["n_unverified"] += 1
        for event in group:
            eid = event["event_id"]
            if eid not in kins:
                continue
            kin, flags = kins[eid]
            metrics = explain_metrics(kin)
            chunk = session["chunks"][event["chunk_id"]]
            if "explain" in force:
                out = scope.step(
                    "explain", EXPLAIN_MODEL,
                    {"clip_id": eid, "metrics": metrics},
                    lambda: {"clip_id": eid, "model": None,
                             "metrics_sent": metrics,
                             "error": "explain_unavailable",
                             "fallback": "respaldo", "detail": "forced"},
                    frames=_clip_frames(kin, chunk))
            else:
                out = scope.step(
                    "explain", None,
                    {"clip_id": eid, "metrics": metrics},
                    lambda e=eid, m=metrics: explain_tool(
                        e, m, session, task_id),
                    frames=_clip_frames(kin, chunk))
            if isinstance(out, dict) and out.get("error"):
                if out.get("error") == "budget_exhausted":
                    counts["n_unverified"] += 1
                    del kins[eid]
                    continue
                explain, explain_error = None, True
            else:
                explain, explain_error = out, False
                scope.steps[-1]["model"] = out.get("model")
            info[eid] = {"kinematics": True, "flags": flags,
                         "explain": explain, "explain_error": explain_error,
                         "segment_clean": False, "event": kin}
        for event in group:
            eid = event["event_id"]
            if eid not in info:
                continue
            have = info[eid]
            hard = any(have["flags"].get(c) is not None for c in HARD_FLAGS)
            explain = have["explain"]
            fp_high = bool(explain and explain.get("false_positive")
                           and explain.get("confidence") == "high")
            if not (hard or fp_high):
                continue
            concept = have["event"]["second"]["cls"]
            concepts.add(concept)
            seg = scope.step(
                "segment", "sam-3.1", {"clip_id": eid, "concept": concept},
                lambda e=eid, c=concept: segment_tool(
                    e, c, session, task_id))
            if isinstance(seg, dict) and not seg.get("error") \
                    and seg.get("masks_visible", 0) > 0 \
                    and seg.get("empty_masks", 0) == 0 \
                    and seg.get("dropped", 0) == 0:
                have["segment_clean"] = True
        for event in group:
            eid = event["event_id"]
            if eid not in info:
                continue
            have = info[eid]
            status, codes = decide(have["flags"], have["explain"],
                                   have["segment_clean"])
            info[eid]["verdict"] = (status, codes)
            verdicts.append({"event_id": eid, "status": status,
                             "reason_codes": codes})
        scope_verdicts = [v for v in verdicts
                          if v["event_id"] in {e["event_id"] for e in group}]
        submit_errors = check_submit([v["event_id"] for v in scope_verdicts],
                                     scope_verdicts,
                                     {k: v for k, v in info.items()
                                      if k in {e["event_id"] for e in group}})
        scope.step("submit", None, {"verdicts": scope_verdicts},
                   lambda: {"errors": submit_errors})
        if submit_errors:
            raise RuntimeError("; ".join(submit_errors))
        steps.extend(scope.steps)
    by_id = {v["event_id"]: v for v in verdicts}
    for event in session["events"]:
        if event["event_id"] in by_id:
            verdict = by_id[event["event_id"]]
            event["status"] = verdict["status"]
            event["verify"] = {
                "reason_codes": verdict["reason_codes"],
                "tool_calls": sorted({s["tool"] for s in steps
                                      if event["event_id"] in _event_names(s)}),
            }
    payload = {"events": session["events"]} if session["events_wrapped"] \
        else session["events"]
    with open(os.path.join(run_dir, "events.json"), "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(run_dir, "trace.jsonl"), "w") as fh:
        for step in steps:
            fh.write(json.dumps(step, ensure_ascii=False) + "\n")
    counts["n_candidates"] = len(verdicts) + counts["n_unverified"]
    counts["n_verified"] = sum(1 for v in verdicts
                               if v["status"] == "verified")
    counts["n_rejected"] = sum(1 for v in verdicts
                               if v["status"] == "rejected")
    counts["n_concepts"] = len(concepts)
    for v in verdicts:
        for code in v["reason_codes"]:
            by_reason[code] = by_reason.get(code, 0) + 1
    counts["cost_usd"] = round(sum(s["cost_usd"] for s in steps), 6)
    counts["rejected_by_reason"] = by_reason
    return counts


def _read_step_results(run_dir):
    """{step_id: result} from every steps/ file under a session run."""
    out = {}
    base = os.path.join(run_dir, "steps")
    if not os.path.isdir(base):
        return out
    for conv_id in sorted(os.listdir(base)):
        conv = os.path.join(base, conv_id)
        if not os.path.isdir(conv):
            continue
        for name in sorted(os.listdir(conv)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(conv, name)) as fh:
                out[f"{conv_id}/{name[:-5]}"] = json.load(fh)
    return out


def _read_explains(run_dir):
    """Successful explain outcomes per event, from the steps files."""
    out = {}
    results = _read_step_results(run_dir)
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        steps = [json.loads(line) for line in fh if line.strip()]
    for step in steps:
        if step["tool"] != "explain":
            continue
        result = results.get(step["step_id"], {})
        if isinstance(result, dict) and not result.get("error"):
            eid = step["args"].get("clip_id") \
                or step["args"].get("event_id")
            if eid:
                out[eid] = result
    return out


def _step_model(run_dir, tool):
    """Model of the first step of ``tool`` in the session trace, or None."""
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            step = json.loads(line) if line.strip() else {}
            if step.get("tool") == tool and step.get("model"):
                return step["model"]
    return None


def _agent_model(run_dir):
    """Model of the first agent turn in the session trace, or None."""
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            step = json.loads(line) if line.strip() else {}
            if step.get("tool") is None and step.get("model"):
                return step["model"]
    return None


def _trace_cost(run_dir):
    """Verify-pass spend from the session trace."""
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.exists(path):
        return 0.0
    with open(path) as fh:
        return round(sum(json.loads(line).get("cost_usd", 0.0)
                         for line in fh if line.strip()), 6)


def _read_concepts(run_dir):
    """Distinct segment concepts from the session trace."""
    path = os.path.join(run_dir, "trace.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        steps = [json.loads(line) for line in fh if line.strip()]
    return {s["args"].get("concept") for s in steps
            if s["tool"] == "segment" and s["args"].get("concept")}


def context(crossing_dir, cf, run_ctx, clusters):
    """Render/validation context: every session's events plus namespaces.

    E-* covers all sessions under the crossing (ids are unique per
    crossing); clearance comes from counterfactual.json when present.
    """
    events = {}
    for name in sorted(os.listdir(crossing_dir)):
        path = os.path.join(crossing_dir, name, "events.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            raw = json.load(fh)
        for event in raw["events"] if isinstance(raw, dict) else raw:
            events[event["event_id"]] = event
    return {"counterfactual": dict(cf["results"],
                                   pick_lpi_s=cf["pick_lpi_s"]),
            "clearance": {c.get("crosswalk_id", c.get("crosswalk")): c
                          for c in cf.get("clearance", [])},
            "run": run_ctx, "metrics": {},
            "clusters": {c["cluster_id"]: c for c in clusters},
            "statuses": {eid: e["status"] for eid, e in events.items()},
            "crossing_dir": crossing_dir,
            **events}


def run_write(run_dir, task_id="T-WRITE", replay=False):
    """Write pass: counterfactual tool steps, then findings + trace.

    Cause and type counts come from explain results in steps/; with no
    explain output they stay empty and top_cause is no determinada.
    """
    session = load_session(run_dir)
    crossing = os.path.dirname(run_dir.rstrip(os.sep))
    replay = replay or os.environ.get("SURI_OFFLINE") == "1"
    for event in session["events"]:
        event["crosswalk"] = session["crosswalks"].get(
            event["conflict_area"])
    cf = counterfactual_doc(session["events"])
    if not os.path.exists(os.path.join(run_dir, "phases.csv")):
        # No signal phases: an LPI is not applicable to this crossing.
        cf.update(signalized=False, pick_lpi_s=0,
                  pick_ref="counterfactual.baseline")
    with open(os.path.join(crossing, "counterfactual.json"), "w") as fh:
        json.dump(cf, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    scope = _Scope(run_dir, "write", task_id, 12, replay)
    for lpi_s in (3, 5):
        scope.step("counterfactual", None, {"lpi_s": lpi_s},
                   lambda k=lpi_s: counterfactual_tool(k, crossing))
    hours = _union_hours(list(session["chunks"].values()))
    verified = [e for e in session["events"] if e["status"] == "verified"]
    clusters = fill_clusters(build_clusters(verified, hours),
                             _read_explains(run_dir))
    causes_all = [c for cluster in clusters for c in cluster["causes"]]
    ordered = list(dict.fromkeys(causes_all))
    top = max(ordered, key=causes_all.count) if ordered \
        else "no determinada"
    by_reason = {}
    for event in session["events"]:
        for code in event.get("verify", {}).get("reason_codes", []):
            by_reason[code] = by_reason.get(code, 0) + 1
    sessions = sorted({session["session_id"]})
    run_ctx = {
        "mode": "replay" if replay else "batch",
        "agent_model": _agent_model(run_dir),
        "explain_model": _step_model(run_dir, "explain") or EXPLAIN_MODEL,
        "crossing_id": os.path.basename(os.path.normpath(crossing)),
        "fallbacks": ["no es en vivo"] if replay else [],
        "hours_observed": hours, "n_candidates": len(session["events"]),
        "n_verified": len(verified),
        "n_rejected": sum(1 for e in session["events"]
                           if e["status"] == "rejected"),
        "n_unverified": sum(1 for e in session["events"]
                            if e["status"] == "candidate"),
        "n_serious_verified": sum(1 for e in verified
                                  if e["tier"] == "serious"),
        "n_concepts": len(_read_concepts(run_dir)),
        "cost_usd": round(sum(s["cost_usd"] for s in scope.steps)
                          + _trace_cost(run_dir), 6),
        "rejected_by_reason": by_reason, "top_cause": top}
    doc = build_findings(clusters, cf, run_ctx, sessions)
    errors = validate(doc, context(crossing, cf, doc["run"], clusters))
    scope.step("submit", None,
               {"findings": [f["finding_id"] for f in doc["findings"]],
                "recommendations": [r["recommendation_id"]
                                    for r in doc["recommendations"]]},
               lambda: {"errors": errors})
    if errors:
        raise RuntimeError("; ".join(errors))
    with open(os.path.join(crossing, "findings.json"), "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    session_name = os.path.basename(os.path.normpath(run_dir))
    with open(os.path.join(crossing, "trace.jsonl"), "w") as fh:
        for step in scope.steps:
            # result_ref is relative to this crossing-level file: the step files live in the session folder.
            out = dict(step, result_ref=f"{session_name}/{step['result_ref']}") if step.get("result_ref") else step
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return {"counterfactual": os.path.join(crossing, "counterfactual.json"),
            "findings": os.path.join(crossing, "findings.json"),
            "trace": os.path.join(crossing, "trace.jsonl"),
            "n_findings": len(doc["findings"]),
            "n_recommendations": len(doc["recommendations"])}


def run_check():
    """Offline contract check over fixtures/agent."""
    import shutil
    import tempfile

    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    joined = _stream_tool_calls([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1",
             "function": {"name": "report_conflict",
                          "arguments": "{\"con"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {
                "arguments": "flict\": true, \"confidence\": \"high\"}"}}]}}]},
    ])
    check("stream tool call",
          joined[0]["function"]["name"] == "report_conflict"
          and json.loads(joined[0]["function"]["arguments"])["conflict"] is True)

    # Table replay: every PET row decides its assigned verdict.
    with open("fixtures/agent/events.json") as fh:
        table = [json.loads(line) for line in fh if line.strip()]
    for row in table:
        flags = {c: None for c in FLAG_CODES}
        flags.update({k: v for k, v in row["flags"].items()
                      if v is not None and k in FLAG_CODES})
        explain = None
        if row["explain"].get("error"):
            info = {"kinematics": True, "flags": flags, "explain": None,
                    "explain_error": True, "segment_clean": False}
        else:
            explain = row["explain"]
            info = {"kinematics": True, "flags": flags, "explain": explain,
                    "explain_error": False,
                    "segment_clean": row["segment_clean"]}
        got = decide(flags, explain, row["segment_clean"])
        check(f"table {row['event_id']}",
              got == (row["expected_status"], row["expected_codes"]))
        check(f"submit {row['event_id']}",
              check_submit([row["event_id"]],
                           [{"event_id": row["event_id"],
                             "status": row["expected_status"],
                             "reason_codes": row["expected_codes"]}],
                           {row["event_id"]: info}) == [])
    # Session replay: the recorded sessions decide their verdicts.
    with open("fixtures/agent/session.jsonl") as fh:
        sessions = [json.loads(line) for line in fh if line.strip()]
    for record in sessions:
        flags = {c: None for c in FLAG_CODES}
        flags.update({k: v for k, v in record["flags"].items()
                      if v is not None and k in FLAG_CODES})
        seg = record["segment"]
        clean = bool(seg and seg["masks_visible"] > 0
                     and seg["empty_masks"] == 0 and seg["dropped"] == 0)
        fp = record["session"] in ("shadow", "L2")
        explain = {"false_positive": fp,
                   "confidence": "high" if fp else None}
        got = decide(flags, explain, clean)
        check(f"session {record['session']}",
              got == (record["expected_verdict"], record["expected_codes"]))
        steps = set(record["required_steps"])
        if record["session"] in ("shadow", "L2"):
            check(f"steps {record['session']}",
                  steps == {"kinematics", "explain", "segment"})
        else:
            check(f"steps {record['session']}",
                  steps == {"kinematics", "explain"})
    # Validator negatives.
    ctx = {"F": {"n_events": 2}, "counterfactual": {}, "run": {},
           "metrics": {}}
    check("V1 digit", validate_text("f", "PET 1.1 s", "es", ctx)
          == ["f: digit"])
    check("V2 number word",
          validate_text("f", "dos encuentros", "es", ctx)
          == ["f: number_word dos"])
    check("V3 unresolved",
          validate_text("f", "{{F.missing}}", "es", ctx)
          == ["f: unresolved {{F.missing}}"])
    check("V3 unknown event",
          validate_text("f", "{{E-9999.pet_s}}", "es", ctx)
          == ["f: unresolved {{E-9999.pet_s}}"])
    check("V1 numeral replace",
          validate_text("f", FINDING_TEMPLATE.replace("{{F.n_events}}", "2",
                                                      1), "es",
                        {"F": {"n_events": 2, "conflict_area": "X",
                               "n_serious": 1, "pet_min_s": 0.5}})
          == ["f: digit"])
    check("V4 claim",
          validate_text("f", "will reduce crashes", "en", ctx)
          == ["f: claim will reduce"])
    check("V4 metrics skipped",
          validate_text("f", "{{metrics.pet_s}}", "es", ctx) == [])
    check("fmt float", fmt_value("pet_s", 2.345) == "2.3")
    check("fmt percent", fmt_value("precision", 0.5) == "50%")
    check("fmt missing", fmt_value("x", None) == "—")
    check("C1 scope",
          check_submit(["E-1"], [], {}) == [
              "C1: verdicts do not match the candidate scope"])
    bad_info = {"E-1": {"kinematics": False, "flags": {}, "explain": None}}
    check("C2 kinematics",
          check_submit(["E-1"], [{"event_id": "E-1", "status": "verified",
                                  "reason_codes": []}], bad_info)
          == ["C2: E-1 no kinematics step"])
    hard_info = {"E-1": {"kinematics": True,
                         "flags": {"id_switch": "first"}, "explain": None,
                         "explain_error": True, "segment_clean": False}}
    check("C3 table",
          check_submit(["E-1"], [{"event_id": "E-1", "status": "verified",
                                  "reason_codes": []}], hard_info)
          == ["C3: E-1 verified against the table"])
    check("C4 code",
          check_submit(["E-1"], [{"event_id": "E-1", "status": "rejected",
                                  "reason_codes": ["nope"]}], hard_info)
          == ["C4: E-1 bad codes ['nope']"])
    all_clear = [{"pet_s": 2.0, "tier": "serious"}]
    check("pick baseline clears all",
          counterfactual_doc(all_clear)["pick_lpi_s"] == 0)
    check("no recs at pick 0",
          build_findings([], counterfactual_doc(all_clear),
                         {}, ["s-check"])["recommendations"] == [])
    check("pick argmin",
          counterfactual_doc(
              [{"pet_s": 0.5, "tier": "serious"},
               {"pet_s": 1.2, "tier": "close"}])["pick_lpi_s"] == 3)
    check("fmt lo", fmt_value("h3.precision_lo", 0.456) == "46%")
    check("fmt money", fmt_value("cost_usd", 0.046) == "0.05")
    groups = cluster_events([{"event_id": "E-1", "window_s": [0.0, 4.0]},
                             {"event_id": "E-2", "window_s": [3.0, 7.0]},
                             {"event_id": "E-3", "window_s": [9.0, 12.0]}])
    check("clusters overlap",
          [[e["event_id"] for e in g] for g in groups]
          == [["E-1", "E-2"], ["E-3"]])
    # End to end offline: verify then write over the fixture session.
    tmp = tempfile.mkdtemp(prefix="agent-check-")
    run_dir = os.path.join(tmp, "s-check")
    shutil.copytree("fixtures/agent/session_files", run_dir)
    with open("fixtures/agent/session_files/events.json") as fh:
        before = {e["event_id"]: e for e in json.load(fh)["events"]}
    os.environ["SURI_OFFLINE"] = "1"
    os.environ["SURI_LEDGER"] = os.path.join(tmp, "ledger.csv")
    os.environ["SURI_CACHE_DIR"] = os.path.join(tmp, "cache")
    counts = run_verify(run_dir, "T-CHECK")
    check("e2e counts", (counts["n_candidates"], counts["n_verified"],
                         counts["n_rejected"], counts["n_unverified"],
                         counts["cost_usd"]) == (2, 2, 0, 0, 0.0))
    with open(os.path.join(run_dir, "trace.jsonl")) as fh:
        steps = [json.loads(line) for line in fh if line.strip()]
    check("trace fields",
          all(sorted(s) == sorted(TRACE_FIELDS) for s in steps))
    check("trace convs",
          sorted({s["step_id"].split("/")[0] for s in steps})
          == ["s-check-c00", "s-check-c01"])
    check("trace tools",
          sorted(s["tool"] for s in steps)
          == ["explain", "explain", "kinematics", "kinematics",
              "submit", "submit"])
    check("trace result_refs",
          all(os.path.exists(os.path.join(run_dir, s["result_ref"]))
              for s in steps))
    results = _read_step_results(run_dir)
    check("explain fallback",
          all(results[s["step_id"]].get("error") == "explain_unavailable"
              and results[s["step_id"]].get("fallback") == "no es en vivo"
              and s["fallback"] == "no es en vivo"
              for s in steps if s["tool"] == "explain"))
    check("submit clean",
          all(results[s["step_id"]] == {"errors": []}
              for s in steps if s["tool"] == "submit"))
    live = os.path.join(tmp, "live")
    os.makedirs(os.path.join(live, "steps", "conv"))
    with open(os.path.join(live, "trace.jsonl"), "w") as fh:
        fh.write(json.dumps({
            "step_id": "conv/s01", "ts": "t", "model": "m",
            "tool": "explain", "args": {"event_id": "E-1"},
            "result_ref": "steps/conv/s01.json", "latency_ms": 1,
            "tokens_in": None, "tokens_out": None, "frames": 80,
            "cost_usd": 0.0, "cache_hit": True,
            "fallback": None}) + "\n")
    with open(os.path.join(live, "steps", "conv", "s01.json"), "w") as fh:
        json.dump({"conflict_type": "through",
                   "likely_cause": "peatón en espera"}, fh)
    check("explains by event_id",
          _read_explains(live) == {
              "E-1": {"conflict_type": "through",
                      "likely_cause": "peatón en espera"}})
    with open(os.path.join(run_dir, "events.json")) as fh:
        updated = json.load(fh)["events"]
    check("events verified",
          all(e["status"] == "verified" for e in updated))
    check("events tool calls",
          all(sorted(e["verify"]["tool_calls"]) == ["explain", "kinematics"]
              for e in updated))
    check("events only status+verify",
          all({k: v for k, v in e.items() if k not in ("status", "verify")}
              == {k: v for k, v in before[e["event_id"]].items()
                  if k not in ("status", "verify")} for e in updated))
    clips = os.path.join(tmp, "clips")
    os.makedirs(clips)
    for eid in ("E-9001", "E-9002"):
        open(os.path.join(clips, f"{eid}.mp4"), "w").close()
    with open(os.path.join(run_dir, "phases.csv"), "w") as fh:
        fh.write("session_id,crosswalk_id,cycle_idx,walk_start_s\n")     # signalized fixture
    out = run_write(run_dir)
    check("write outputs", (out["n_findings"], out["n_recommendations"])
          == (2, 1))
    with open(os.path.join(tmp, "counterfactual.json")) as fh:
        cf = json.load(fh)
    check("counterfactual scenario",
          cf["scenario"] == "escenario mecánico")
    check("counterfactual pick", cf["pick_lpi_s"] == 3)
    with open(os.path.join(tmp, "trace.jsonl")) as fh:
        wtrace = [json.loads(line) for line in fh if line.strip()]
    check("write trace",
          sorted(s["tool"] for s in wtrace)
          == ["counterfactual", "counterfactual", "submit"])
    with open(os.path.join(tmp, "findings.json")) as fh:
        doc = json.load(fh)
    check("findings shape",
          doc["crossing_id"] == os.path.basename(tmp)          # from the run folder, not a constant
          and doc["sessions"] == ["s-check"]
          and doc["run"]["mode"] == "replay"
          and doc["run"]["fallbacks"] == ["no es en vivo"]
          and doc["references"] == REFERENCES
          and all("evidence" in f for f in doc["findings"]))
    with open(os.path.join(run_dir, "events.json")) as fh:
        events = json.load(fh)["events"]
    verified = [e for e in events if e["status"] == "verified"]
    clusters = build_clusters(verified, 20.0 / 3600.0)
    vctx = context(tmp, cf, doc["run"], clusters)
    check("findings validate", validate(doc, vctx) == [])
    fctx = dict(vctx, F=vctx["clusters"]["F-01"])
    check("finding renders",
          render(doc["findings"][0]["text_template"], fctx)
          == "Zona de conflicto XW-S. Encuentros verificados: 1. "
             "Serios: 1. PET mínimo: 0.5 s.")
    check("rec renders",
          render(doc["recommendations"][0]["text_template"], vctx)
          == "Se recomienda evaluar un intervalo peatonal adelantado "
             "(LPI) de 3 s para los cruces peatonales. En el escenario "
             "mecánico, los conflictos serios candidatos pasarían de 1 a 0.")
    check("summary renders",
          render(doc["summary"]["es"], vctx)
          == "Se verificaron candidatos con conflicto entre peatón y "
             "vehículo. La causa más frecuente fue no determinada. Se "
             "propone como medida un intervalo peatonal adelantado. La "
             "decisión queda en manos del equipo de ingeniería.")
    nosig = build_findings(clusters, dict(cf, signalized=False, pick_lpi_s=0),
                           doc["run"], ["s-check"])
    check("no signal, no LPI",
          nosig["recommendations"] == []
          and "no tiene semáforo" in nosig["summary"]["es"]
          and "intervalo peatonal adelantado. La decisión" in nosig["summary"]["es"]
          and all(f["evidence"]["clips"] and f["evidence"]["keyframes"]
                  for f in nosig["findings"]))
    bad_ctx = dict(vctx, crossing_dir=os.path.join(tmp, "empty"))
    check("V5 missing clip",
          validate(doc, bad_ctx)
          == ["F-01: missing clip clips/E-9001.mp4",
              "F-02: missing clip clips/E-9002.mp4"])
    try:
        _blurred_clip({"run_dir": run_dir}, "E-9001")
        tag_rejected = False
    except ValueError:
        tag_rejected = True
    check("clip tag gate", tag_rejected)
    check("templates numeral-free",
          validate_text("t", FINDING_TEMPLATE, "es", fctx) == []
          and validate_text("t", doc["recommendations"][0]["text_template"],
                            "es", vctx) == []
          and validate_text("t", ES_TEMPLATE, "es", vctx) == []
          and validate_text("t", EN_TEMPLATE, "en", vctx) == [])
    shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("agent --check: PASS (table, sessions, validator, e2e, write)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wonder verification agent")
    sub = parser.add_subparsers(dest="command")
    verify = sub.add_parser("verify", help="verify candidates in RUN_DIR")
    verify.add_argument("run_dir")
    verify.add_argument("--task", required=True)
    verify.add_argument("--replay", action="store_true")
    verify.add_argument("--force", default="",
                        help="comma list from {explain}")
    write = sub.add_parser("write", help="write findings for RUN_DIR")
    write.add_argument("run_dir")
    write.add_argument("--task", default="T-WRITE")
    write.add_argument("--replay", action="store_true")
    val = sub.add_parser("validate", help="validate a findings.json file")
    val.add_argument("findings")
    val.add_argument("--run", default=None,
                     help="session run (defaults to doc sessions[0])")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return run_check()
    if args.command == "verify":
        counts = run_verify(args.run_dir, args.task, replay=args.replay,
                            force={f for f in args.force.split(",") if f})
        print(json.dumps(counts, indent=2, ensure_ascii=False))
        return 0
    if args.command == "write":
        out = run_write(args.run_dir, task_id=args.task, replay=args.replay)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        with open(args.findings) as fh:
            doc = json.load(fh)
        crossing = os.path.dirname(os.path.abspath(args.findings))
        with open(os.path.join(crossing, "counterfactual.json")) as fh:
            cf = json.load(fh)
        session_name = os.path.basename(args.run.rstrip(os.sep)) \
            if args.run else doc["sessions"][0]
        run_dir = os.path.join(crossing, session_name)
        session = load_session(run_dir)
        for event in session["events"]:
            event["crosswalk"] = session["crosswalks"].get(
                event["conflict_area"])
        verified = [e for e in session["events"]
                    if e["status"] == "verified"]
        clusters = build_clusters(verified)
        errors = validate(doc, context(crossing, cf, doc["run"], clusters))
        for error in errors:
            print("FAIL " + error)
        if errors:
            return 1
        print("findings validate: PASS")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
