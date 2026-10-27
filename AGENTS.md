# AGENTS.md — Working rules for Muse agents (Suri)

This repo is the implementation of Suri for the Meta Global AI Developer
Hackathon. Suri watches one crossing, measures pedestrian–vehicle close calls
(post-encroachment time, PET), verifies each candidate with an agent, and
produces a video-assisted road-safety inspection. The code computes, the AI
explains, the engineer decides.

## Language

Repository language is English. Code, comments, docs, and commit messages are
in English. (Conversation with the repo owner may be in Spanish; repo content
never is.)

## Repo map (planned)

| Path | What |
| --- | --- |
| `src/ingest.py` | Re-encode to 10 fps, ROI crop, chunk split → `chunks.csv` |
| `src/sam.py`, `src/ledger.py` | Segmentation client, stream parser, response cache, cost ledger |
| `src/calib.py` | Pixel-to-meter homography, `calib.json`, `zones.json`, `zones.png` |
| `src/kinematics.py` | Trajectories, speeds, `crossings.csv`, `events.json` (PET) |
| `src/counterfactual.py` | Leading-pedestrian-interval scenario → `counterfactual.json` |
| `src/agent.py` | Verification agent, `trace.jsonl`, `findings.json` |
| `src/metrics.py` | Precision/recall and cost metrics → `metrics.json` |
| `src/clips.py` | Blurred 720p evidence clips + keyframes, `--audit` |
| `src/report.py` | HTML + PDF inspection report (Spanish + English summary) |
| `app.py`, `docs/` | Local live run + static site |
| `fixtures/<module>/` | One fixture per module, used by its check |
| `runs/` | `ledger.csv`, per-crossing and per-session outputs (see below) |

## TDD — the general rule

- Every module ships with a fixture under `fixtures/<module>/` and one runnable
  check that proves its contract against that fixture.
- Write the check first, watch it fail, then implement.
- All checks run with one command. A change is done only when its check and
  every neighboring check pass in this session.
- A confident-looking patch that was never run against the repo checks is not
  done. Re-run the checks after every fix, and also exercise edge and error
  paths (empty/None/malformed input, overlaps, boundary values).

## Data contracts (minimal, binding)

Run layout:

- `runs/ledger.csv`: the project-wide cost ledger.
- `runs/<crossing_id>/`: crossing-level outputs (`eval.json`,
  `ground_truth.csv`, `counterfactual.json`, `metrics.json`, `findings.json`,
  `trace.jsonl`, `clips/`).
- `runs/<crossing_id>/<session_id>/`: session-level outputs (`chunks.csv`,
  `tracks.csv`, `calib.json`, `zones.json`, `phases.csv`, `crossings.csv`,
  `events.json`).

IDs:

- Crossing: `mty-garciaroel-ahuehuetes`. Sessions: `g1-20261027`, `g2-20261030`.
- Chunks: `<session_id>-cNN`. Events: `E-<session#><nnn>`, unique per
  crossing (G1 uses E-1001…, G2 uses E-2001…).

Key rules:

- `tracks.csv` columns are fixed; `X_m`/`Y_m` are filled by kinematics only.
- Events carry `pet_s`, `tier` (`serious` if PET < 1.0 s, `close` if
  1.0 ≤ PET < 3.0 s; pairs ≥ 3.0 s are not emitted), `status`
  (`candidate`/`verified`/`rejected`), and `verify.reason_codes`.
- `counterfactual.json` always carries `"scenario": "escenario mecánico"`.
- Every API response is cached by request hash; spend goes through the ledger.

## Verification-agent rules (for `src/agent.py`)

- Four tools only: `kinematics(event_id)`, `segment(clip_id, concept)`,
  `explain(clip_id, metrics)`, `counterfactual(lpi_s)`.
- Agent prose contains no numerals — no digits, no number words. Numbers enter
  only through placeholders (e.g. `{{E-1001.pet_s}}`) resolved from tool
  outputs. A validator fails any numeral outside placeholders and references.
- Kinematics outranks `explain` on numbers; `explain` informs conflict type and
  rationale only.
- Rejection reason codes: `id_switch`, `mask_jump`, `impossible_speed`,
  `occlusion`, `paths_do_not_cross`, `parked_or_static`, `explain_disagrees`,
  `outside_phase_logic`.
- Every finding cites ≥ 1 event and its clip. Every recommendation cites ≥ 1
  finding cluster, the counterfactual result, and a guideline reference.
- Fallbacks are labeled: model fallback as `respaldo`, replayed runs as
  `no es en vivo`. The report ends with an engineer-decision box; the agent
  never claims a change *will* reduce crashes.

## Privacy (binding)

- Raw video never enters this repo and is deleted on Nov 30, 2026.
- Only blurred material is published or sent to generative models: faces and
  plates blurred, 720p, no audio, metadata stripped.
- No face recognition, no plate reading, no enforcement use — never.
- Model APIs are called in Standard mode only, never Contributor/training mode.

## Honesty

- Never claim first, only, predicts crashes, validated speeds, any video,
  real time, or most dangerous.
- Every number in reports, pages, and video traces to a tool output or a cited
  source measured during the event.

## Delivery workflow

- Never commit, amend, push, tag, or publish. Leave all work staged
  (`git add <named files>`) for manual review.
- Keep units small: one behavior per change, with its check evidence stated.
- Verify before handing off: run the repo checks covering the touched area and
  quote the result. If a check fails on touched code, the task is not done.
- After editing, confirm no absolute local paths or machine-specific
  references leaked into repo files.
