"""Crossing metrics: rates and costs from tool outputs (PRD FR-13).

Computes what the recorded evidence supports: observed hours (chunk
overlaps counted once), the verified-serious rate with a normal
approximation to the Poisson interval, ledger cost per video minute
(H6), and a copy of the counterfactual clearance. Every hypothesis
that needs hand annotation stays null and is listed in unscored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import agent


def _read_events(run_dir):
    path = os.path.join(run_dir, "events.json")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        raw = json.load(fh)
    return raw["events"] if isinstance(raw, dict) else raw


def _read_chunks(run_dir):
    path = os.path.join(run_dir, "chunks.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _ledger_total(ledger_path):
    if not os.path.exists(ledger_path):
        return 0.0
    with open(ledger_path, newline="") as fh:
        return round(sum(float(r.get("cost_usd") or 0.0)
                         for r in csv.DictReader(fh)), 6)


def _null_block(keys):
    return {key: (0 if key == "n" or key.endswith("_n") else None)
            for key in keys}


def compute(crossing_dir, ledger_path):
    """metrics.json content from the crossing's recorded evidence."""
    sessions = sorted(name for name in os.listdir(crossing_dir)
                      if os.path.isdir(os.path.join(crossing_dir, name))
                      and os.path.exists(
                          os.path.join(crossing_dir, name, "events.json")))
    events = [e for name in sessions
              for e in _read_events(os.path.join(crossing_dir, name))]
    chunks = [c for name in sessions
              for c in _read_chunks(os.path.join(crossing_dir, name))]
    hours = agent._union_hours(chunks)
    footage_min = sum(agent._chunk_end(c) - float(c["t0_s"]) for c in chunks)
    footage_min = round(footage_min / 60.0, 3)
    verified = [e for e in events if e.get("status") == "verified"]
    n_serious = sum(1 for e in verified if e.get("tier") == "serious")
    if hours > 0 and n_serious:
        rate = n_serious / hours
        half = 1.96 * math.sqrt(n_serious) / hours
        rate_block = {"rate_per_h": round(rate, 1),
                      "rate_per_h_lo": round(max(0.0, rate - half), 1),
                      "rate_per_h_hi": round(rate + half, 1),
                      "n": n_serious}
    else:
        rate_block = {"rate_per_h": 0.0 if hours > 0 else None,
                      "rate_per_h_lo": None, "rate_per_h_hi": None,
                      "n": n_serious}
    total_usd = _ledger_total(ledger_path)
    h6_value = round(total_usd / footage_min, 2) if footage_min else None
    cf_path = os.path.join(crossing_dir, "counterfactual.json")
    clearance = []
    if os.path.exists(cf_path):
        with open(cf_path) as fh:
            clearance = json.load(fh).get("clearance", [])
    unscored = []
    if not os.path.exists(os.path.join(crossing_dir, "ground_truth.csv")):
        unscored.append("ground_truth.csv missing: h1, h3, h4, h5, "
                        "pet_error and agreement stay null")
    if not os.path.exists(os.path.join(crossing_dir, "eval.json")):
        unscored.append("eval.json missing: h2 stays null")
    return {
        "hours_observed": hours,
        "serious": rate_block,
        "h1": _null_block(("value", "value_lo", "value_hi", "n")),
        "h2": _null_block(("value", "value_lo", "value_hi", "n")),
        "h3": _null_block(("precision", "precision_lo", "precision_hi",
                            "precision_n", "recall", "recall_lo",
                            "recall_hi", "recall_n", "f1", "f1_lo",
                            "f1_hi", "f1_n")),
        "h4": _null_block(("precision_before", "precision_after",
                            "recall_lost", "recall_lost_lo",
                            "recall_lost_hi", "n")),
        "h5": _null_block(("value", "value_lo", "value_hi", "n")),
        "h6": {"value": h6_value, "value_lo": None, "value_hi": None,
               "ledger_usd": total_usd, "footage_min": footage_min},
        "pet_error": _null_block(("mae_s", "mae_s_lo", "mae_s_hi", "n")),
        "agreement": None,
        "clearance": clearance,
        "unscored": unscored,
    }


def run_check():
    """Offline contract check over fixtures/metrics."""
    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    doc = compute("fixtures/metrics/crossing", "fixtures/metrics/ledger.csv")
    check("hours union", doc["hours_observed"] == 0.014)
    check("serious rate", doc["serious"] == {
        "rate_per_h": 142.9, "rate_per_h_lo": 0.0,
        "rate_per_h_hi": 340.8, "n": 2})
    check("h6", doc["h6"]["value"] == 0.06
          and doc["h6"]["ledger_usd"] == 0.06
          and doc["h6"]["footage_min"] == 1.0)
    check("nulls", doc["h3"]["precision"] is None
          and doc["h3"]["precision_n"] == 0
          and doc["pet_error"]["mae_s"] is None
          and doc["agreement"] is None)
    check("clearance copy", doc["clearance"] == [
        {"crosswalk": "XW-S", "share_in_crosswalk_at_flash_end": 0.2}])
    check("unscored", len(doc["unscored"]) == 2)
    check("fmt rate", agent.fmt_value("serious.rate_per_h", 111.14)
          == "111.1")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("metrics --check: PASS (rates, h6, clearance, nulls)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write metrics.json")
    parser.add_argument("crossing_dir", nargs="?")
    parser.add_argument("--ledger",
                        default=os.path.join("runs", "ledger.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return run_check()
    doc = compute(args.crossing_dir, args.ledger)
    out = args.out or os.path.join(args.crossing_dir, "metrics.json")
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
