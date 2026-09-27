"""HTML offline do caso real: entrada, RAG, demonstracoes, payloads e metricas."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads((args.run / "trace.json").read_text())
    trace["artifacts_path"] = str(args.run)
    trace["api_artifacts"] = {
        p.name: json.loads(p.read_text()) for p in sorted((args.run / "api").glob("*.json"))
    }
    payload = (
        json.dumps(trace, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    template = Path("scripts/reports/templates/exemplo-completo.html").read_text()
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/exemplo-completo.html").write_text(
        template.replace("__TRACE_DATA__", payload)
    )
    print("outputs/reports/exemplo-completo.html")


if __name__ == "__main__":
    main()
