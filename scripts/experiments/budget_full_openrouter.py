"""Orcamento offline por tokens reais do dev e tarifas publicas salvas; sem API."""

import json
from pathlib import Path


def tariff(endpoint, prompt_tokens):
    price = dict(endpoint["pricing"])
    for override in sorted(price.get("overrides", []), key=lambda x: x["min_prompt_tokens"]):
        if prompt_tokens >= override["min_prompt_tokens"]:
            price.update(override)
    return float(price["prompt"]), float(price["completion"])


def main():
    root = Path("outputs/dev10-complete-20260926")
    out = Path("outputs/full-planning-20260926")
    items = []
    mapping = [
        ("ling-free", "ling-paid", "deepinfra/fp4"),
        ("qwen37", "qwen37", "alibaba"),
        ("gemma26", "gemma26", "darkbloom"),
    ]
    for source, target, tag in mapping:
        catalog = json.loads((out / f"{target}-endpoints.json").read_text())
        endpoint = next(e for e in catalog["response"]["data"]["endpoints"] if e["tag"] == tag)
        responses = [
            json.loads(p.read_text()) for p in (root / "run" / source).glob("*.response.json")
        ]
        assert len(responses) == 60
        cost = ceiling = 0
        for r in responses:
            u = r["usage"]
            a, b = tariff(endpoint, u["prompt_tokens"])
            cost += u["prompt_tokens"] * a + u["completion_tokens"] * b
            ceiling += u["prompt_tokens"] * a + 8192 * b
        items.append(
            {
                "item": target,
                "generation_calls": 6000,
                "estimated_usd": cost * 100,
                "same_input_all_outputs_8192_usd": ceiling * 100,
                "pricing_source": catalog["url"],
                "retrieved_at": catalog["retrieved_at"],
            }
        )
    catalog = json.loads((out / "qwen37-endpoints.json").read_text())
    endpoint = next(e for e in catalog["response"]["data"]["endpoints"] if e["tag"] == "alibaba")
    probes = list((root / "qwen-prepared/qwen37/calibration").glob("*.response.json"))
    cost = 0
    for p in probes:
        usage = json.loads(p.read_text())["usage"]
        a, b = tariff(endpoint, usage["prompt_tokens"])
        cost += usage["prompt_tokens"] * a + usage["completion_tokens"] * b
    items.append(
        {
            "item": "qwen-calibration",
            "estimated_calls": len(probes) * 100,
            "estimated_usd": cost * 100,
            "same_input_all_outputs_8192_usd": cost * 100,
        }
    )
    total = sum(i["estimated_usd"] for i in items)
    stress = sum(i["same_input_all_outputs_8192_usd"] for i in items)
    report = {
        "basis": "dev10 corrected source x100; no cache discounts; NOT full preflight",
        "ling_paid_is_budget_scenario_not_automatic_migration": True,
        "items": items,
        "total_usd": total,
        "total_plus_30pct_usd": total * 1.3,
        "recommended_reserve_usd": 60,
        "output_stress_scenario_usd": stress,
        "output_stress_plus_30pct_usd": stress * 1.3,
        "generation_calls": 18000,
        "projected_probe_calls": len(probes) * 100,
        "new_generation_calls": 0,
        "new_probe_calls": 0,
        "notes": [
            "Snapshot tariffs; no taxes/purchase fees/FX/local compute.",
            "Retries and changed length distributions can exhaust caps; no completion guarantee.",
            "Projected probes scale from 10 documents; actual shared cache can change count.",
        ],
    }
    (out / "budget.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
