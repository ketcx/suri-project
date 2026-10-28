"""Shared API cache, cost ledger and per-model caps.

Every API call in the project goes through ``call``:
cache lookup, cap check, send, atomic cache write, ledger row.

Conventions kept here on purpose:

- ``request_hash`` is the sha256 of the canonical JSON of
  ``{"endpoint": ..., "body": key_body}``. Media enters the body only as a
  media key (chunk sha256, or ``<sha>:<f0>-<f1>:<variant>`` for a window),
  so a re-upload or a re-cut still hits the cache.
- ``call`` injects its ``model`` argument into the key body before
  hashing, so the same endpoint and body under another model is a miss.
- A cache entry is ``{"request": ...}`` on line 1, then one JSON object per
  event in received order. It is written to ``.tmp`` and renamed only after
  the stream completes, so a broken stream never becomes an entry.
- ``SURI_OFFLINE=1`` turns a cache miss into ``CacheMiss`` (checks and the
  demo replay). ``cache_reads=False`` skips the lookup but still writes.
- Paths come from the environment so checks can redirect them:
  ``SURI_LEDGER`` (default ``runs/ledger.csv``),
  ``SURI_CACHE_DIR`` (the ``api/`` directory holding ``<hash>.jsonl.gz``).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone

PRICES = {
    # Per 1,000 frames, min 50 billed per request.
    "sam-3.1": {"per_1000_frames": 0.20, "min_frames": 50},
    # Per 1M input / cached input / output tokens.
    "muse-spark-1.3": {"per_m_in": 1.25, "per_m_cached": 0.15, "per_m_out": 4.25},
    "muse-wonder-1.0": {"per_m_in": 4.00, "per_m_cached": 0.20, "per_m_out": 20.00},
    # Per image.
    "muse-image-1.0": {"per_image": 0.01},
    # Per hour of audio.
    "muse-voice-transcribe-1.0": {"per_hour": 0.18},
    # Baseten (Muse Spark Open). Exact model id is confirmed in T-101;
    # rates are the secondary-source values from the spec until then.
    "baseten/muse-spark-open": {
        "per_m_in": 1.25,
        "per_m_cached": 0.15,
        "per_m_out": 4.25,
    },
}

CAPS = {
    "sam-3.1": 45.0,
    "muse-wonder-1.0": 50.0,
    "muse-spark-1.3": 10.0,
    "muse-image-1.0": 2.0,
    "muse-voice-transcribe-1.0": 1.0,
    "baseten": 100.0,
}

LEDGER_COLUMNS = (
    "ts",
    "task_id",
    "model",
    "endpoint",
    "request_hash",
    "frames",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "cache_hit",
)


class CapExceeded(Exception):
    """Raised before anything is sent when a call would pass its cap."""


class CacheMiss(Exception):
    """Raised on a miss when SURI_OFFLINE=1 instead of sending."""


class TruncatedStream(Exception):
    """Raised when no attempt ends with the closing event."""


def _is_complete(events: list) -> bool:
    """Judge completion on the whole event list, not one event.

    - Responses streams (events carry "type"): complete only if the last
      type is the closing event ("response.completed"; "completed" is
      accepted for the offline fixture).
    - Chat-completion chunk streams: complete if some chunk carries a
      non-null choices[].finish_reason.
    - Anything else must be exactly one object (a non-streaming 2xx):
      complete unless it carries a "status" other than "completed".
    """
    if not events or not all(isinstance(e, dict) for e in events):
        return False
    if any("type" in e for e in events):
        return events[-1].get("type") in ("response.completed", "completed")
    if any(e.get("object") == "chat.completion.chunk" for e in events):
        return any(
            isinstance(c, dict) and c.get("finish_reason") is not None
            for e in events
            for c in (e.get("choices") or [])
        )
    if len(events) == 1:
        return events[0].get("status", "completed") == "completed"
    return False


def _ledger_path() -> str:
    return os.environ.get("SURI_LEDGER", os.path.join("runs", "ledger.csv"))


def _cache_dir() -> str:
    path = os.environ.get("SURI_CACHE_DIR")
    if not path:
        raise ValueError("SURI_CACHE_DIR is not set")
    return path


def _cap_for(model: str) -> float:
    if model.startswith("baseten/"):
        return CAPS["baseten"]
    try:
        return CAPS[model]
    except KeyError:
        raise ValueError(f"no cap configured for model {model!r}") from None


def request_hash(endpoint: str, key_body: dict) -> str:
    """sha256 of the canonical JSON of endpoint + body (with media keys)."""
    canonical = json.dumps(
        {"endpoint": endpoint, "body": key_body},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def spent_usd(model: str) -> float:
    """Sum of cost_usd for model over the ledger. Missing file means 0."""
    path = _ledger_path()
    if not os.path.exists(path):
        return 0.0
    total = 0.0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") == model:
                total += float(row.get("cost_usd") or 0.0)
    return total


def _append_row(row: dict) -> None:
    path = _ledger_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fresh = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in LEDGER_COLUMNS})


def _entry_path(request_h: str) -> str:
    return os.path.join(_cache_dir(), request_h + ".jsonl.gz")


def _read_entry(path: str) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    if not lines or "request" not in lines[0]:
        raise ValueError(f"bad cache entry: {path}")
    return lines[1:]


def _write_entry(path: str, request: dict, events: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        with gzip.open(tmp, "wt") as fh:
            fh.write(json.dumps({"request": request}) + "\n")
            for event in events:
                fh.write(json.dumps(event) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _extract_usage(last: dict) -> dict:
    """Usage from the closing event, top level or nested under response."""
    usage = last.get("usage") or {}
    if not usage and isinstance(last.get("response"), dict):
        usage = last["response"].get("usage") or {}
    return usage


def _cost_from_usage(model: str, usage: dict, estimate_usd: float) -> tuple[float, int, int, int]:
    """Return (cost_usd, frames, tokens_in, tokens_out).

    SAM/image/voice without usable usage fall back to the estimate.
    An LLM call without usage is a data error and raises instead of
    recording 0, which would silently defeat its cap.
    """
    if not usage:
        if "per_m_in" in PRICES.get(model, {}):
            raise ValueError(f"{model}: closing event carries no usage")
        print("ledger: no usage in completed event, using estimate", file=sys.stderr)
        return estimate_usd, 0, 0, 0
    if model == "sam-3.1":
        frames = int(usage.get("video_frames_processed", 0))
        if frames <= 0:
            print("ledger: no video_frames_processed, using estimate", file=sys.stderr)
            return estimate_usd, 0, 0, 0
        billed = max(frames, PRICES["sam-3.1"]["min_frames"])
        return billed * PRICES["sam-3.1"]["per_1000_frames"] / 1000.0, frames, 0, 0
    if model == "muse-image-1.0":
        n = int(usage.get("images", 1))
        return n * PRICES[model]["per_image"], 0, 0, 0
    if model == "muse-voice-transcribe-1.0":
        seconds = float(usage.get("audio_seconds", 0))
        if seconds <= 0:
            return estimate_usd, 0, 0, 0
        return seconds / 3600.0 * PRICES[model]["per_hour"], 0, 0, 0
    rates = PRICES[model]
    t_in = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    t_cached = int(usage.get("cached_tokens", 0))
    if not t_cached:
        for details_key in ("input_tokens_details", "prompt_tokens_details"):
            details = usage.get(details_key) or {}
            t_cached = int(details.get("cached_tokens", 0))
            if t_cached:
                break
    t_out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
    if t_in == 0 and t_out == 0:
        raise ValueError(f"{model}: usage has no known token fields")
    cost = ((t_in - t_cached) * rates["per_m_in"] + t_cached * rates["per_m_cached"] + t_out * rates["per_m_out"]) / 1e6
    return cost, 0, t_in, t_out


def call(task_id, model, endpoint, key_body, send, estimate_usd=0.0, cache_reads=True):
    """Cache lookup, cap check, send, atomic cache write, ledger row.

    Yields event dicts in stored order on a hit, so downstream parsing is
    byte-identical. ``send`` is called at most twice: a stream that breaks
    after sending gets a row at the estimate and 1 retry.

    The ``model`` argument is part of the cache identity: it is injected
    into the key body before hashing, so switching models never replays
    another model's response. An uncached send requires a positive,
    finite ``estimate_usd`` so failed attempts always leave spend behind.
    """
    if model.endswith("-contributor"):
        raise ValueError("Contributor mode is never used")
    if not isinstance(key_body, dict):
        raise ValueError(f"key_body must be a dict, got {type(key_body).__name__}")
    if "model" in key_body and key_body["model"] != model:
        raise ValueError(f"key_body model {key_body['model']!r} does not match {model!r}")
    key_body = {**key_body, "model": model}
    request_h = request_hash(endpoint, key_body)
    request = {"endpoint": endpoint, "body": key_body}
    path = _entry_path(request_h)
    if cache_reads and os.path.exists(path):
        for event in _read_entry(path):
            yield event
        _append_row(
            {
                "ts": _now_iso(),
                "task_id": task_id,
                "model": model,
                "endpoint": endpoint,
                "request_hash": request_h,
                "frames": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0,
                "cache_hit": 1,
            }
        )
        return
    if os.environ.get("SURI_OFFLINE") == "1":
        raise CacheMiss(f"offline and not cached: {request_h}")
    try:
        estimate = float(estimate_usd)
    except (TypeError, ValueError):
        raise ValueError(f"{model}: estimate_usd must be a positive USD amount, got {estimate_usd!r}") from None
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError(f"{model}: estimate_usd must be a positive, finite USD amount, got {estimate_usd!r}")
    def log_estimate() -> None:
        _append_row(
            {
                "ts": _now_iso(),
                "task_id": task_id,
                "model": model,
                "endpoint": endpoint,
                "request_hash": request_h,
                "frames": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": estimate_usd,
                "cache_hit": 0,
            }
        )

    # One retry: every broken attempt gets its own row at the estimate,
    # and the cap is rechecked before every send, since the failed
    # attempt's estimate already counts toward it.
    cap = _cap_for(model)
    broken = 0
    events: list = []
    while True:
        spent = spent_usd(model)
        if spent >= cap or spent + estimate_usd > cap:
            raise CapExceeded(f"{model}: spent {spent:.4f} + est {estimate_usd:.4f} >= cap {cap}; budget exhausted")
        try:
            events = list(send())
        except Exception:  # noqa: BLE001 - logged, retried once, then propagated
            log_estimate()
            if broken >= 1:
                raise
            broken += 1
            continue
        if not _is_complete(events):
            log_estimate()
            if broken >= 1:
                raise TruncatedStream(f"no closing event after retry: {request_h}")
            broken += 1
            continue
        break
    usage = _extract_usage(events[-1])
    try:
        cost_usd, frames, tokens_in, tokens_out = _cost_from_usage(model, usage, estimate_usd)
    except ValueError:
        log_estimate()
        raise
    # The call already happened and was billed, so the spend row goes
    # first: a cache I/O failure must not erase it from the ledger.
    _append_row(
        {
            "ts": _now_iso(),
            "task_id": task_id,
            "model": model,
            "endpoint": endpoint,
            "request_hash": request_h,
            "frames": frames,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "cache_hit": 0,
        }
    )
    _write_entry(path, request, events)
    yield from events


def _check() -> None:
    """Replay is identical, caps stop calls, no Contributor.

    Covers the cache, the cap, truncated streams with retry, nested
    usage pricing, the LLM-without-usage error, per-model cache
    identity, the spent-at-cap block, and the estimate requirement.
    """
    import shutil

    tmp = tempfile.mkdtemp(prefix="suri-ledger-check-")
    os.environ["SURI_LEDGER"] = os.path.join(tmp, "ledger.csv")
    os.environ["SURI_CACHE_DIR"] = os.path.join(tmp, "api")
    try:
        events = [
            {"type": "delta", "text": "<0f>1<|box;x1=10;y1=20;x2=30;y2=60;w=320;h=240|>"},
            {"type": "delta", "text": "<1f>1<|box;x1=12;y1=20;x2=32;y2=60;w=320;h=240|>"},
            {"type": "completed", "usage": {"video_frames_processed": 300}},
        ]
        body = {"model": "sam-3.1", "input": [{"concept": "person"}]}
        got = list(call("T-103", "sam-3.1", "meta/responses", body, lambda: iter(events), estimate_usd=0.06))
        assert got == events, "replay must be identical"
        again = list(call("T-103", "sam-3.1", "meta/responses", body, lambda: (_ for _ in ()).throw(AssertionError("send on hit"))))
        assert again == events, "cache hit must replay identically"

        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2, rows
        assert rows[0]["cache_hit"] == "0" and float(rows[0]["cost_usd"]) > 0, rows[0]
        assert rows[0]["frames"] == "300", rows[0]
        assert rows[1]["cache_hit"] == "1" and float(rows[1]["cost_usd"]) == 0, rows[1]

        def boom():
            raise AssertionError("send must not run past the cap")

        old = CAPS["sam-3.1"]
        CAPS["sam-3.1"] = 0.0
        try:
            try:
                list(call("T-103", "sam-3.1", "meta/responses", {"other": True}, boom, estimate_usd=0.06))
            except CapExceeded:
                pass
            else:
                raise AssertionError("CapExceeded not raised")
        finally:
            CAPS["sam-3.1"] = old

        try:
            list(call("T-103", "sam-3.1-contributor", "meta/responses", body, boom))
        except ValueError:
            pass
        else:
            raise AssertionError("contributor model not refused")

        # Truncated stream: first attempt has no closing event, retry wins.
        # The ledger keeps one row at the estimate plus the success row.
        truncated = [
            {"type": "delta", "text": "<0f>1<|box;x1=1;y1=2;x2=3;y2=4;w=320;h=240|>"},
            {"type": "error", "message": "stream cut"},
        ]
        sends = iter([truncated, events])
        fixed = {"concept": "person", "attempt": "retry"}
        got = list(call("T-103", "sam-3.1", "meta/responses", fixed, lambda: next(sends), estimate_usd=0.06))
        assert got == events, "retry must yield the complete stream"
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4, rows
        assert float(rows[2]["cost_usd"]) == 0.06 and rows[2]["cache_hit"] == "0", rows[2]
        assert float(rows[3]["cost_usd"]) > 0, rows[3]

        # Two truncated attempts raise instead of caching a partial stream.
        sends = iter([truncated, truncated])
        try:
            list(call("T-103", "sam-3.1", "meta/responses", {"retry": "twice"}, lambda: next(sends), estimate_usd=0.06))
        except TruncatedStream:
            pass
        else:
            raise AssertionError("TruncatedStream not raised")
        assert not os.path.exists(os.path.join(os.environ["SURI_CACHE_DIR"], request_hash("meta/responses", {"retry": "twice", "model": "sam-3.1"}) + ".jsonl.gz"))

        # Nested usage (Responses API shape) prices an LLM call correctly.
        nested = [
            {"type": "delta", "text": "verifying E-1001"},
            {"type": "completed", "response": {"usage": {"input_tokens": 200000, "output_tokens": 50000}}},
        ]
        list(call("T-203", "muse-wonder-1.0", "meta/responses", {"agent": True}, lambda: iter(nested), estimate_usd=0.01))
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            cost = float(list(csv.DictReader(fh))[-1]["cost_usd"])
        assert abs(cost - 1.80) < 1e-9, cost

        # An LLM call without usage fails instead of recording 0.
        nousage = [{"type": "completed"}]
        try:
            list(call("T-203", "muse-wonder-1.0", "meta/responses", {"empty": True}, lambda: iter(nousage), estimate_usd=0.01))
        except ValueError:
            pass
        else:
            raise AssertionError("LLM without usage not rejected")

        # A non-streaming chat completion is cached after 1 send, with cost.
        chat = [
            {
                "object": "chat.completion",
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            }
        ]
        calls = []

        def send_chat():
            calls.append(1)
            return iter(chat)

        got = list(call("T-202", "muse-spark-1.3", "baseten/chat/completions", {"chat": True}, send_chat, estimate_usd=0.01))
        assert got == chat and len(calls) == 1, (got, len(calls))
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            cost = float(list(csv.DictReader(fh))[-1]["cost_usd"])
        assert abs(cost - 0.001675) < 1e-12, cost
        again = list(call("T-202", "muse-spark-1.3", "baseten/chat/completions", {"chat": True}, send_chat))
        assert again == chat and len(calls) == 1, "non-streaming call must hit the cache"

        # A chunk stream ending in finish_reason is cached after 1 send.
        chunks = [
            {"object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "verified"}, "finish_reason": "stop"}]},
            {"object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 50}},
        ]
        got = list(call("T-202", "muse-wonder-1.0", "meta/chat/completions", {"chunks": True}, lambda: iter(chunks), estimate_usd=0.01))
        assert got == chunks, "chunk stream must be cached after 1 send"
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            cost = float(list(csv.DictReader(fh))[-1]["cost_usd"])
        assert abs(cost - 0.003) < 1e-12, cost
        again = list(call("T-202", "muse-wonder-1.0", "meta/chat/completions", {"chunks": True}, lambda: iter(chunks)))
        assert again == chunks, "chunk replay must be identical"

        # A lone response with a non-completed status is never cached.
        lone = [{"object": "response", "status": "incomplete"}]
        try:
            list(call("T-202", "muse-wonder-1.0", "meta/responses", {"lone": True}, lambda: iter(lone), estimate_usd=0.01))
        except TruncatedStream:
            pass
        else:
            raise AssertionError("incomplete response not rejected")
        assert not os.path.exists(os.path.join(os.environ["SURI_CACHE_DIR"], request_hash("meta/responses", {"lone": True, "model": "muse-wonder-1.0"}) + ".jsonl.gz"))

        # Unknown token field names fail instead of pricing at 0.
        weird = [{"type": "completed", "response": {"usage": {"input_token_count": 200000, "output_token_count": 50000}}}]
        try:
            list(call("T-203", "muse-wonder-1.0", "meta/responses", {"weird": True}, lambda: iter(weird), estimate_usd=0.01))
        except ValueError:
            pass
        else:
            raise AssertionError("unknown usage fields not rejected")

        # Cached tokens are read from the nested details object.
        detail = [
            {
                "type": "completed",
                "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200}, "output_tokens": 100},
            }
        ]
        list(call("T-202", "muse-wonder-1.0", "meta/responses", {"detail": True}, lambda: iter(detail), estimate_usd=0.01))
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            cost = float(list(csv.DictReader(fh))[-1]["cost_usd"])
        assert abs(cost - 0.00524) < 1e-12, cost

        # A key body naming another model is refused instead of cached.
        try:
            list(call("T-202", "muse-spark-1.3", "baseten/chat/completions", {"model": "sam-3.1"}, boom, estimate_usd=0.01))
        except ValueError:
            pass
        else:
            raise AssertionError("model mismatch not rejected")

        # The model is part of the cache identity: the same endpoint
        # and body under another model is a miss, not a replay.
        spark_events = [
            {"type": "delta", "text": "spark answer"},
            {"type": "completed", "usage": {"input_tokens": 100, "output_tokens": 10}},
        ]
        wonder_events = [
            {"type": "delta", "text": "wonder answer"},
            {"type": "completed", "usage": {"input_tokens": 100, "output_tokens": 10}},
        ]
        spark_sends, wonder_sends = [], []

        def send_spark():
            spark_sends.append(1)
            return iter(spark_events)

        def send_wonder():
            wonder_sends.append(1)
            return iter(wonder_events)

        got = list(call("T-202", "muse-spark-1.3", "meta/responses", {"iso": True}, send_spark, estimate_usd=0.01))
        assert got == spark_events, "spark send must yield its own events"
        got = list(call("T-203", "muse-wonder-1.0", "meta/responses", {"iso": True}, send_wonder, estimate_usd=0.01))
        assert got == wonder_events, "wonder must not replay the spark response"
        assert spark_sends == [1] and wonder_sends == [1], (spark_sends, wonder_sends)
        spark_h = request_hash("meta/responses", {"iso": True, "model": "muse-spark-1.3"})
        wonder_h = request_hash("meta/responses", {"iso": True, "model": "muse-wonder-1.0"})
        assert spark_h != wonder_h, "cache identities must differ across models"
        assert os.path.exists(os.path.join(os.environ["SURI_CACHE_DIR"], spark_h + ".jsonl.gz"))
        assert os.path.exists(os.path.join(os.environ["SURI_CACHE_DIR"], wonder_h + ".jsonl.gz"))

        # Spending exactly at the cap blocks a live call, not just
        # estimates that overshoot it; the send never runs.
        at_cap_sends = []

        def send_at_cap():
            at_cap_sends.append(1)
            return iter(events)

        old_spark_cap = CAPS["muse-spark-1.3"]
        CAPS["muse-spark-1.3"] = spent_usd("muse-spark-1.3")
        try:
            try:
                list(call("T-202", "muse-spark-1.3", "baseten/chat/completions", {"at-cap": True}, send_at_cap, estimate_usd=0.01))
            except CapExceeded:
                pass
            else:
                raise AssertionError("CapExceeded not raised when spent == cap")
        finally:
            CAPS["muse-spark-1.3"] = old_spark_cap
        assert at_cap_sends == [], "send ran after the budget was exhausted"

        # An uncached send needs a positive, finite estimate, so a
        # failed attempt always leaves spend behind; send never runs.
        for bad_est in (0.0, -0.01, float("nan"), float("inf"), None):
            bad_sends = []

            def send_bad():
                bad_sends.append(1)
                return iter(events)

            try:
                list(call("T-103", "sam-3.1", "meta/responses", {"bad-est": True}, send_bad, estimate_usd=bad_est))
            except ValueError:
                pass
            else:
                raise AssertionError(f"bad estimate accepted: {bad_est!r}")
            assert bad_sends == [], f"send ran with estimate {bad_est!r}"
        try:
            list(call("T-103", "sam-3.1", "meta/responses", {"no-est": True}, lambda: iter(events)))
        except ValueError:
            pass
        else:
            raise AssertionError("omitted estimate accepted")

        # SURI_OFFLINE=1 raises CacheMiss on a miss without sending.
        os.environ["SURI_OFFLINE"] = "1"
        try:
            try:
                list(call("T-103", "sam-3.1", "meta/responses", {"offline": True}, boom))
            except CacheMiss:
                pass
            else:
                raise AssertionError("CacheMiss not raised")
        finally:
            del os.environ["SURI_OFFLINE"]
        assert not os.path.exists(os.path.join(os.environ["SURI_CACHE_DIR"], request_hash("meta/responses", {"offline": True, "model": "sam-3.1"}) + ".jsonl.gz"))

        # cache_reads=False skips the lookup but still writes the entry.
        fresh = [{"object": "chat.completion", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
        fresh_calls = []

        def send_fresh():
            fresh_calls.append(1)
            return iter(fresh)

        got = list(call("T-202", "muse-spark-1.3", "baseten/chat/completions", {"fresh": True}, send_fresh, cache_reads=False, estimate_usd=0.01))
        assert got == fresh and len(fresh_calls) == 1, (got, len(fresh_calls))
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            assert list(csv.DictReader(fh))[-1]["cache_hit"] == "0"
        got = list(call("T-202", "muse-spark-1.3", "baseten/chat/completions", {"fresh": True}, send_fresh))
        assert got == fresh and len(fresh_calls) == 1, "second call must hit the cache"

        # A retry that no longer fits the cap raises before the second send.
        old_cap = CAPS["sam-3.1"]
        CAPS["sam-3.1"] = spent_usd("sam-3.1") + 0.09
        flaky_calls = []

        def send_flaky():
            flaky_calls.append(1)
            if len(flaky_calls) == 1:
                return iter(truncated)
            return iter(events)

        try:
            try:
                list(call("T-103", "sam-3.1", "meta/responses", {"flaky": True}, send_flaky, estimate_usd=0.06))
            except CapExceeded:
                pass
            else:
                raise AssertionError("CapExceeded not raised before retry")
        finally:
            CAPS["sam-3.1"] = old_cap
        assert len(flaky_calls) == 1, flaky_calls

        # A cache-write failure still leaves the billed charge in the ledger.
        blocker = os.path.join(tmp, "file-as-dir")
        with open(blocker, "w") as fh:
            fh.write("x")
        os.environ["SURI_CACHE_DIR"] = blocker
        try:
            try:
                list(call("T-103", "sam-3.1", "meta/responses", {"uncacheable": True}, lambda: iter(events), estimate_usd=0.06))
            except OSError:
                pass
            else:
                raise AssertionError("cache failure not raised")
        finally:
            os.environ["SURI_CACHE_DIR"] = os.path.join(tmp, "api")
        with open(os.environ["SURI_LEDGER"], newline="") as fh:
            row = list(csv.DictReader(fh))[-1]
        assert row["cache_hit"] == "0" and float(row["cost_usd"]) == 0.06, row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ledger --check: PASS (replay, cap, contributor, truncation, usage)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        _check()
    else:
        sys.exit("usage: python src/ledger.py --check")
