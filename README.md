# Suri 🐾

**Suri watches a crossing, measures its close calls and gives the city evidence to act on.**

Suri takes its name from the meerkat (*suricata*), the sentinel that watches from high ground and warns the group before danger arrives.

> **Status (Nov 4, 2026):** the pipeline runs end to end, from video to a verified Spanish inspection report, on one open research clip. Monterrey footage is the next step. This is an entry in the **Meta Global AI Developer Hackathon** (Indie track). All work here was created between Oct 26 and Nov 4, 2026.

**Report:** [`docs/index.html`](docs/index.html) (open it in a browser; served by GitHub Pages once Pages points at `/docs`). **Trace of the agent:** [`runs/murcia-s1/g2-20261104/trace.jsonl`](runs/murcia-s1/g2-20261104/trace.jsonl).

## Why

Cities usually act on a dangerous crossing only after people are hurt. At García Roel and Ahuehuetes, a crossing in Distrito Tec, Monterrey, INEGI open data records 16 crashes in 2019–2024 and no pedestrian hit. Crash records only count what already went wrong. Near misses happen far more often than crashes, and they can be measured from video.

## How it works

1. **Track.** Meta **SAM 3.1** tracks people and cars (`person`, `car`) from a text prompt, one concept per request, with no training and no labels.
2. **Measure.** Code computes the **post-encroachment time (PET)** per conflict zone: the seconds between one road user leaving the zone and the other entering it. Under 3 s is a close call; under 1 s it is serious.
3. **Verify.** A **Muse Wonder** agent sees JSON only, never video. It calls `kinematics` (plausibility flags computed by code) and `explain`, where **Muse Spark** watches the blurred clip and names a likely cause. It then submits a verdict that code re-checks against fixed rules, so a model can never override a measurement.
4. **Report.** The output is a Spanish *"Inspección de seguridad vial asistida por video"*. Code fills in every number, people and plates are blurred, and the last box is for the engineer's decision.

**The code computes, the AI explains, the engineer decides.**

## Demo results (open clip, not Monterrey)

The input is the roadside-camera video of scenario 1 of the Universidad de Murcia's *Multi-perspective Traffic Video Recording* (CC BY 4.0): 2.7 minutes, run at 10 fps in six 30-second chunks.

| Step | Result |
| --- | --- |
| SAM 3.1 tracking | People and cars tracked through all 162 s |
| PET candidates (code) | 7 in one conflict zone: 5 serious, 2 close |
| Muse Wonder verification | 2 tool-calling turns: `kinematics` ×7 and `explain` ×7 in parallel, then `submit_verdict` ×7; every verdict passed the code re-check |
| Verdicts | **4 verified** (3 serious, 1 close) and **3 rejected** (`paths_do_not_cross`: the two paths never came within the threshold inside the event window) |
| Muse Spark causes | 6 of 7 clips explained, for example "no cede el paso" (the car does not yield) and "peatón esperando" (pedestrian waiting); one call timed out after 3 attempts, so that verdict rests on the code flags |
| Cost | USD 2.80 for all 126 API calls made while building, including tests and re-runs; USD 0.59 per minute of footage (`runs/murcia-s1/metrics.json`). The final live verification run cost USD 0.17 |

**What we tried first.** We also ran the same scenario's drone video. From that height SAM 3.1 saw the cars but not the pedestrians, so it produced no PET pairs; its tracks are kept in `runs/murcia-s1/g1-20261104/`. A fixed roadside camera, which is Suri's design, worked.

## Meta models used

| Model | Role |
| --- | --- |
| SAM 3.1 (`sam-3.1`) | Tracks people and cars in video (Responses API, streamed) |
| Muse Wonder (`muse-wonder-1.0`) | Verification agent: calls tools and submits verdicts that code re-checks |
| Muse Spark 1.3 (`muse-spark-1.3`) | `explain` tool: watches each blurred 720p clip; fallback agent model |
| Muse Code | Wrote most pipeline modules under the rules in [AGENTS.md](AGENTS.md) |

Meta API notes from building this:
- Stream every call (non-streaming calls hung).
- `tool_choice` takes only `"auto"`.
- Keep tool schemas flat.
- Send video through the Responses API (`chat.completions` rejects `input_video`).

## Repo map

| Path | What |
| --- | --- |
| `src/ingest.py` | Re-encode to 10 fps, crop, split into chunks → `chunks.csv` |
| `src/sam.py`, `src/ledger.py` | SAM 3.1 client and stream parser; response cache and cost ledger |
| `src/calib.py` | Pixel-to-meter homography and zone checks; this demo uses zones in pixels |
| `src/kinematics.py` | Tracks, zone occupancy and PET → `events.json` |
| `src/verify_live.py` | Live Muse Wonder verification: Responses API, streaming, flat tools, code re-check |
| `src/agent.py` | Findings, clusters, mechanical scenario and the numbers validator → `findings.json` |
| `src/clips.py` | Blurred 720p evidence clips and keyframes, with `--audit` |
| `src/report.py` | Spanish HTML report → `docs/index.html` |
| `src/metrics.py` | Rates and cost → `metrics.json`. Precision and recall stay null, because there is no manual annotation yet |
| `fixtures/<module>/` | Data for each module's offline check |
| `runs/` | `ledger.csv` and the outputs of each run |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # MODEL_API_KEY; never commit it
make checks                   # every module check, offline
```

To reproduce the demo from cache (needs the raw clip in `raw/murcia/`; it is not in the repo):

```bash
export SURI_CACHE_DIR=runs/murcia-s1/cache/api
python3 src/kinematics.py runs/murcia-s1/g2-20261104
python3 src/clips.py runs/murcia-s1/g2-20261104
python3 src/verify_live.py runs/murcia-s1/g2-20261104 --task T-202
python3 src/agent.py write runs/murcia-s1/g2-20261104
python3 src/report.py runs/murcia-s1/g2-20261104
```

## Limitations (honest)

- One staged clip from Spain. The Monterrey crossing has not been recorded yet.
- PET is not yet validated against manual frame-by-frame annotation.
- There is no metric calibration in this demo, so plausibility flags are in pixels and there are no speeds.
- The demo crossing has no traffic signal, so no leading pedestrian interval is proposed.

## Privacy

- Raw video is never committed.
- Published clips and keyframes are 720p with every person and every whole car box blurred (a partial blur left a high-mounted plate readable, so spec 09's fallback applies), and carry the `blurred-v1` tag. Only tagged clips are ever sent to a model.
- Suri does no face recognition and no plate reading.
- Model calls use Standard mode, never Contributor mode.

*Aviso de privacidad (ES): el video crudo nunca se sube al repositorio. Solo se publica material difuminado, sin caras ni placas legibles. No se usa reconocimiento facial ni lectura de placas.*

## Built during the event

- All code, data and media in this repo were created between **Oct 26 and Nov 4, 2026**. The commit history is the record.
- **Prior work, disclosed:** the author has built road-safety software before, an Android app called *Vial-IA*. No code, data or assets from it are used here.

## Attributions

- Video data: Sanchez-Iborra, R., Asensio-Garriga, R., Alarcon-Hellin, G., Bernal-Escobedo, L., Trevlakis, S., Tsiftsis, T., & Skarmeta, A. (2026). *Multi-perspective Traffic Video Recording* (v1) [Data set]. Zenodo. https://zenodo.org/records/18375218. License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Scenario 1, RSU camera (`RSU_1_no_fall.MOV`). Changes: resampled to 10 fps, people and plates blurred, boxes and zones overlaid by Suri.
- Crash data: INEGI, ATUS and the georeferenced municipal crash base 2019–2024, used under INEGI's free-use terms.
- Built with SAM 3.1, Muse Wonder, Muse Spark and Muse Code (Meta).

## Team

Armando Medina (solo): from the Dominican Republic, MSc Data Science exchange in Monterrey, Mexico.
