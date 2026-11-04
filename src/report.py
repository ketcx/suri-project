"""Inspection report: single-file HTML from verified findings (muse-03).

Reads one session run plus its crossing-level findings.json and
counterfactual.json, resolves the agent templates, and writes a
standalone docs page with the event table, blurred keyframes, the
agent text, and the engineer-decision box. Numbers come only from
the inputs; the fixed prose carries no claims about prediction,
real time, or crash reduction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import agent

BANNED = ("predic", "tiempo real", "reducir", "reducirá", "avalad",
          "predict", "real-time", "real time", "will reduce", "endorse")

HEADER = "Inspección de seguridad vial asistida por video: demostración del método"

LIMITATIONS = (
    "Escenario controlado de otro país.",
    "Un solo clip de una cámara fija.",
    "PET sin validar contra anotación manual.",
    "Sin velocidades métricas: el análisis usa píxeles.",
)

def decision_box(has_recommendation):
    """Engineer sign-off. An LPI checkbox only exists when one was proposed."""
    if has_recommendation:
        first = ("Se acepta la recomendación y se estudia el intervalo "
                 "peatonal adelantado.")
    else:
        first = "Se acepta el informe tal como está."
    return f"""<section>
<h2>La persona decide</h2>
<p>El método propone; la decisión es del equipo de ingeniería.</p>
<ul>
<li><input type="checkbox"> {first}</li>
<li><input type="checkbox"> Se rechaza: la evidencia no basta.</li>
<li><input type="checkbox"> Se pide más evidencia (más sesiones, anotación manual del PET).</li>
</ul>
<p>Firma: ____________________ Fecha: __________</p>
</section>"""


def load(run_dir, atus_path):
    """Events, findings, counterfactual, and ATUS context as dicts."""
    session = agent.load_session(run_dir)
    crossing = os.path.dirname(run_dir.rstrip(os.sep))
    with open(os.path.join(crossing, "findings.json")) as fh:
        findings = json.load(fh)
    with open(os.path.join(crossing, "counterfactual.json")) as fh:
        counterfactual = json.load(fh)
    with open(atus_path) as fh:
        atus = json.load(fh)
    return {"session": session, "findings": findings,
            "counterfactual": counterfactual, "atus": atus,
            "crossing": crossing}


REPLAY_BANNER = ("<p><strong>Repetición sin modelos en vivo:</strong> "
                   "esta página se generó solo con la evidencia guardada, "
                   "sin llamar a ningún modelo.</p>")


def resolve(data):
    """Render every agent template with its validation context."""
    session, findings = data["session"], data["findings"]
    cf = data["counterfactual"]
    crossing = os.path.dirname(session["run_dir"].rstrip(os.sep))
    for event in session["events"]:
        event["crosswalk"] = session["crosswalks"].get(
            event["conflict_area"])
    verified = [e for e in session["events"] if e["status"] == "verified"]
    clusters = agent.fill_clusters(
        agent.build_clusters(verified),
        agent._read_explains(session["run_dir"]))
    ctx = agent.context(crossing, cf, findings["run"], clusters)
    texts = []
    for finding in findings["findings"]:
        fctx = dict(ctx, F=ctx["clusters"][finding["cluster_id"]])
        texts.append((finding["finding_id"],
                      agent.render(finding["text_template"], fctx)))
    recs = [(r["recommendation_id"],
             agent.render(r["text_template"], ctx))
            for r in findings["recommendations"]]
    summary = agent.render(findings["summary"]["es"], ctx)
    return {"texts": texts, "recs": recs, "summary": summary,
            "clusters": clusters}


def render_html(data, resolved, clip_prefix="clips"):
    """Standalone HTML page. Event rows carry the measured numbers."""
    session, findings = data["session"], data["findings"]
    atus = data["atus"]
    cross = atus["crossing_A"]["within_30m"]
    banner = REPLAY_BANNER if findings["run"].get("fallbacks") else ""
    rows = []
    for event in session["events"]:
        codes = ", ".join(event["verify"]["reason_codes"]) or "—"
        rows.append(
            f"<tr><td>{event['event_id']}</td>"
            f"<td>{event['pet_s']:.1f}</td><td>{event['tier']}</td>"
            f"<td>{event['status']}</td><td>{codes}</td>"
            f"<td><img src=\"{clip_prefix}/{event['event_id']}.jpg\" "
            f"alt=\"keyframe {event['event_id']}\" width=\"320\"></td></tr>")
    agent_paras = "\n".join(
        f"<p><strong>{fid}</strong>: {text}</p>"
        f"<p>Tipos: {', '.join(cluster['conflict_types']) or '—'}; "
        f"causas señaladas: {'; '.join(cluster['causes']) or '—'}.</p>"
        for (fid, text), cluster in zip(resolved["texts"],
                                        resolved["clusters"]))
    rec_paras = "\n".join(f"<p><strong>{rid}</strong>: {text}</p>"
                          for rid, text in resolved["recs"])
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{HEADER}</title>
<style>body{{font-family:sans-serif;max-width:60em;margin:auto;padding:1em}}
table{{border-collapse:collapse}}td,th{{border:1px solid #999;padding:4px}}</style>
</head>
<body>
<h1>{HEADER}</h1>
{banner}
<section>
<h2>Contexto: García Roel y Ahuehuetes, Monterrey</h2>
<p>En el cruce hubo {cross['accidents']} choques en
{atus['crossing_A']['georef_years']} y ningún atropello registrado.
Los registros de choques no ven los casi-atropellos. Este cruce es el
siguiente paso: todavía no se ha grabado.</p>
</section>
<section>
<h2>Resultados de la demostración: video abierto de la Universidad de Murcia (España), no de Monterrey</h2>
<p>{resolved['summary']}</p>
<table>
<tr><th>evento</th><th>PET (s)</th><th>nivel</th><th>veredicto</th>
<th>códigos</th><th>keyframe difuminado</th></tr>
{"".join(rows)}
</table>
</section>
<section>
<h2>Lo que dice el agente</h2>
{agent_paras}
{rec_paras}
</section>
<section>
<h2>Fuentes</h2>
<ul>
<li>Videos: Sanchez-Iborra, R., Asensio-Garriga, R., Alarcon-Hellin,
G., Bernal-Escobedo, L., Trevlakis, S., Tsiftsis, T., &amp; Skarmeta, A.
(2026). Multi-perspective Traffic Video Recording (v1) [Data set].
Zenodo. https://zenodo.org/records/18375218. Licencia CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Escenario 1, cámara RSU
(RSU_1_no_fall.MOV). Cambios: recorte a 10 fps, personas y placas
difuminadas, cajas y zonas superpuestas por Suri.</li>
<li>{atus['attribution']}.</li>
<li>Modelos: SAM 3.1 para segmentación; Muse Wonder para verificar cada
candidato con herramientas (nunca ve video); Muse Spark para explicar el clip
difuminado. Llamadas y costos en el ledger del proyecto.</li>
</ul>
</section>
<section>
<h2>Limitaciones</h2>
<ul>
{"".join(f"<li>{item}</li>" for item in LIMITATIONS)}
</ul>
</section>
{decision_box(bool(resolved["recs"]))}
</body>
</html>
"""


def run_check():
    """Offline contract check over fixtures/report."""
    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    data = load("fixtures/report/run", "fixtures/report/atus.json")
    resolved = resolve(data)
    html = render_html(data, resolved)
    check("header", HEADER in html)
    check("context", "16 choques" in html and "ningún atropello" in html)
    check("table", "E-9001" in html and "0.5" in html and "verified" in html)
    check("agent text", "1 encuentros verificados" in html)
    check("no placeholders", "{{" not in html)
    check("decision box", "La persona decide" in html)
    check("decision box closes the page",
          html.rfind("<h2>La persona decide</h2>")
          > html.rfind("<h2>Limitaciones</h2>"))
    check("no lpi checkbox without a recommendation",
          "se estudia el intervalo peatonal adelantado" not in html)
    rec_findings = dict(data["findings"], recommendations=[{
        "recommendation_id": "R-01",
        "text_template": "Texto de recomendación.",
    }])
    rec_html = render_html(dict(data, findings=rec_findings),
                           resolve(dict(data, findings=rec_findings)))
    check("lpi checkbox with a recommendation",
          "se estudia el intervalo peatonal adelantado" in rec_html)
    check("limitations", all(item in html for item in LIMITATIONS))
    check("sources", "CC BY 4.0" in html and "INEGI" in html)
    check("citation", "zenodo.org/records/18375218" in html
          and "Cambios: recorte a 10 fps" in html
          and "RSU_1_no_fall.MOV" in html)
    lowered = html.lower()
    check("no banned claims",
          all(word not in lowered for word in BANNED))
    replayed = dict(data)
    replayed["findings"] = dict(data["findings"],
                                run=dict(data["findings"]["run"],
                                         fallbacks=["no es en vivo"]))
    check("replay banner",
          "Repetición sin modelos en vivo" in render_html(replayed,
                                                          resolve(replayed)))
    check("causes line", "Tipos:" in html and "causas señaladas" in html)
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("report --check: PASS (header, table, agent text, box, sources)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the HTML report")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--atus", default="docs/data/atus.json")
    parser.add_argument("--out", default="docs/index.html")
    parser.add_argument("--clip-prefix", default="clips")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return run_check()
    data = load(args.run_dir, args.atus)
    html = render_html(data, resolve(data), clip_prefix=args.clip_prefix)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
