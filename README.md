# Suri 🐾

**Suri watches a crossing, measures its close calls and gives the city evidence and a fix to test.**

*Suri vigila un cruce, mide sus casi-accidentes y le da al municipio evidencia y una corrección para probar.*

Suri takes its name from the meerkat (*suricata*), the sentinel that watches from high ground and warns the group before danger arrives.

> **Status (Oct 27, 2026): build starts.**  This repo is the implementation. Part of the **Meta Global AI Developer Hackathon** (Indie track); all work here is new, created Oct 26 – Nov 4, 2026.

## Why

Cities usually act on a dangerous crossing only after people are hurt. Nuevo León records about 1 in 5 of Mexico's urban crashes, yet in 88.7% of its 2023 road deaths the record does not say whether the victim was walking, riding or driving (INEGI ATUS; STCONAPRA 2023–2024). Near misses happen far more often than crashes and can be measured from video. Tools that do this exist in North America and cost thousands of dollars per intersection; we found no deployments in Latin America.

## How it works

1. **Capture.** Fixed, elevated video of one crossing in Distrito Tec, Monterrey (`mty-garciaroel-ahuehuetes`), plus a measured ground network and signal timing (spec 01).
2. **Track.** Meta **SAM 3.1** segments and tracks every `person` and `car` (spec 02).
3. **Measure.** Tracks are projected onto the street plane; the code computes **post-encroachment time (PET)** per conflict area, pedestrian crossing speeds, and who is still in the crosswalk when the signal turns red (specs 03–04).
4. **Verify.** A **Muse Wonder** agent checks every candidate with tools — kinematics, re-segmentation with concepts it writes itself, and an explanation from **Muse Spark Open** — and rejects false alarms with reason codes (spec 06).
5. **Recommend.** The agent proposes a **leading pedestrian interval (LPI 3/5/7 s)** and replays it on the measured tracks as a labeled mechanical scenario, not a prediction (spec 05).
6. **Report.** Output is a Spanish *"Inspección de seguridad vial asistida por video"* where every finding links to its clip, metric and check. **The code computes, the AI explains, the engineer decides.**

## Meta models

| Model | Role |
| --- | --- |
| SAM 3.1 (`sam-3.1`) | Detection, masks and tracking in video |
| Muse Wonder (`muse-wonder-1.0`) | Inspection agent: plans, calls tools, verifies, writes |
| Muse Spark Open (Baseten) | Explains each evidence clip (open weights) |
| Muse Spark 1.3 (`muse-spark-1.3`) | Fallback for the agent and explanations |
| Muse Image (`muse-image-1.0`) | Optional illustration of the proposed fix |
| Muse Voice Transcribe | Optional narration recorded with Ray-Ban Meta glasses |

API access uses the OpenAI-compatible SDK with `base_url=https://api.meta.ai/v1` (Meta) and `https://inference.baseten.co/v1` (Baseten).

## Repo map (planned)

| Path | What | Spec |
| --- | --- | --- |
| `src/ingest.py` | Re-encode to 10 fps, ROI crop, chunk split → `chunks.csv` | 02 |
| `src/sam.py`, `src/ledger.py` | SAM 3.1 client, stream parser, cache + cost ledger | 02 |
| `src/calib.py` | Pixel-to-meter homography (RANSAC), `calib.json`, `zones.json`, `zones.png` | 03 |
| `src/kinematics.py` | Trajectories, speeds, `crossings.csv`, `events.json` (PET) | 04 |
| `src/counterfactual.py` | LPI 3/5/7 s mechanical scenario → `counterfactual.json` | 05 |
| `src/agent.py` | Wonder agent, 4 tools, placeholders-only prose, `trace.jsonl`, `findings.json` | 06 |
| `src/metrics.py` | H1–H6, Wilson/Poisson intervals, `metrics.json` | 07 |
| `src/clips.py` | Blurred 720p evidence clips + keyframes, `--audit` | 09 |
| `src/report.py` | HTML + PDF inspection report (ES + EN summary) | 08 |
| `app.py`, `docs/` | Local live run + static GitHub Pages site | 08 |
| `fixtures/<module>/` | One runnable check per module | 02–07 |
| `runs/` | `ledger.csv`, `<crossing_id>/<session_id>/` outputs (see PRD §7) | PRD |

Full contracts (IDs, CSV columns, JSON schemas) are in the planning repo: `docs/PRD.md` §7 and `docs/specs/`.

## Quickstart (planned)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # MODEL_API_KEY, BASETEN_API_KEY — never commit
python -m src.ingest --help
```

All checks run with one command (see PRD §5, reproducibility). Every API response is cached by request hash; spend goes through `runs/ledger.csv`.

## Privacy

- Raw video stays on the author's machine and is deleted on Nov 30, 2026.
- Published clips are blurred (faces and plates) and downscaled. The street camera records no audio; no face recognition or plate reading is ever used.
- Video goes to the Meta Model API in Standard mode only, never Contributor mode.
- To request removal, contact [contact email].

*Aviso de privacidad (ES): el video crudo se queda en el equipo del autor y se borra el 30 nov 2026. Solo se publica material difuminado, sin caras ni placas legibles y sin audio de la calle. No se usa reconocimiento facial ni lectura de placas.*

## Built during the event

- All code, data and media in this repo are created between **Oct 26 and Nov 4, 2026**. The commit history is the record.
- **Prior work, disclosed:** the author has built road-safety software before, an Android app called *Vial-IA*. No code, data or assets from it are used here.

## Attributions

- Built with Muse Spark Open.
- Fuente: INEGI.
- STCONAPRA, *Informe de salud pública sobre la situación de la seguridad vial, México 2023–2024*.
- WHO, *Global status report on road safety 2023*.
- FHWA-HRT-18-044 (leading pedestrian interval evaluation).

## Team

Armando Medina, AI engineer, solo builder, Monterrey.
