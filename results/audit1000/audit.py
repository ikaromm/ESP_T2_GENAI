"""Audit completed raw generations, unchanged earlier results and the final matrix."""

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "src"))
from scripts.execution.run_paid_concurrent import held_cost, validate_interruption_audit
from scripts.execution.run_prepared_paid import TARGETS, target_for_saved_attempt, validate_response

ROOT = Path.cwd()
OUT = ROOT / "outputs/round-521-1000-20260929"
ARMS = ("C1", "C1t", "C2", "C3", "C4", "C5")
MODELS = (("ling-free", "ling"), ("qwen37", "qwen"), ("gemma26", "gemma"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


lock = json.loads((ROOT / "configs/full_openrouter.lock.json").read_text())
ids = [r["doc_id"] for r in lock["cohort"]]
assert len(ids) == len(set(ids)) == 1000
amendment = json.loads((ROOT / "configs/ling-paid-round-521-1000.json").read_text())
history = json.loads((ROOT / "outputs/full-rounds/history" / f"{amendment['plan_id']}.json").read_text())
assert history["result"]["complete"] and history["plan"]["documents"] == ids[520:]
assert not any(history["result"]["pending_generations"].values())
report = {"checked_at": datetime.now(UTC).isoformat(), "documents": 1000,
          "new_documents": 480, "models": {}, "scientific_changes": False}
for model, slug in MODELS:
    folder = ROOT / f"outputs/full-{slug}-batches" / model
    prepared = ROOT / f"outputs/full-{slug}-prepared"
    rows = json.loads((prepared / model / "preflight.json").read_text())
    ledger = json.loads((folder / "ledger.json").read_text())
    before = json.loads((OUT / "ledger-before" / f"{model}.json").read_text())
    assert ledger["identity"] == before["identity"]
    assert ledger["attempts"][:len(before["attempts"])] == before["attempts"]
    accepted = [a for a in ledger["attempts"] if a["status"] == "accepted"]
    assert len(accepted) == 6000 and {a["case"] for a in accepted} == set(range(6000))
    assert not any(a["status"] == "pending" for a in ledger["attempts"])
    assert [(r["doc_id"], r["arm"]) for r in rows] == [(d,a) for d in ids for a in ARMS]
    for i in range(1000):
        group = rows[i*6:i*6+6]
        assert group[1]["context_tokens"] == group[2]["context_tokens"]
        assert group[1]["prompt_tokens"] == group[2]["prompt_tokens"]
        assert len({r["context"] for r in group[2:]}) == 1
    cost = Decimal(0)
    tokens_in = tokens_out = 0
    finishes = Counter()
    for a in accepted:
        row = rows[a["case"]]
        target = target_for_saved_attempt(TARGETS[model], folder, a, row)
        response = json.loads((folder / a["response_file"]).read_text())
        validate_response(response, target, row)
        if a["case"] >= 3120:
            cost += Decimal(str(response["usage"]["cost"]))
            tokens_in += response["usage"]["prompt_tokens"]
            tokens_out += response["usage"]["completion_tokens"]
            finishes[response["choices"][0]["finish_reason"]] += 1
    for a in ledger["attempts"]:
        if a["status"] == "audited_interruption":
            validate_interruption_audit(a, folder, rows[a["case"]], TARGETS["ling-paid-novita"])
    selected = [a for a in ledger["attempts"] if a["case"] >= 3120]
    held = held_cost(ledger, set(range(3120,6000)))
    assert held <= Decimal(amendment["round_budget_usd"][model])
    report["models"][model] = {
        "accepted_total": 6000, "new_accepted": 2880, "reported_round_cost_usd": str(cost),
        "held_round_cost_usd": str(held), "round_budget_usd": amendment["round_budget_usd"][model],
        "new_input_tokens": tokens_in, "new_output_tokens": tokens_out,
        "new_finish_reasons": dict(finishes),
        "round_attempt_statuses": dict(Counter(a["status"] for a in selected)),
        "ledger_sha256": digest(folder / "ledger.json"),
    }
    print(model, "6000/6000 validated", flush=True)

with (ROOT / "results/audit520/raw-artifact-manifest.csv").open() as handle:
    originals = list(csv.DictReader(handle))
assert len(originals) == 9360
for r in originals:
    for part in ("request", "response"):
        assert digest(ROOT / r[f"{part}_path"]) == r[f"{part}_sha256"]
print("Earlier 9360 raw generations preserved", flush=True)

summary = json.loads((ROOT / "results/progress/summary.json").read_text())
assert summary["documents_compared"] == summary["expected_documents"] == 1000
assert summary["document_ids"] == ids
analysis = json.loads((ROOT / "results/progress/analysis.json").read_text())
assert analysis["documents"] == 1000 and analysis["tests"] == 30
assert analysis["scores_sha256"] == digest(ROOT / "results/progress/scores.csv")
assert analysis["comparisons_sha256"] == digest(ROOT / "results/progress/comparisons.json")
assert analysis["comparisons_csv_sha256"] == digest(ROOT / "results/progress/comparisons.csv")
with (ROOT / "results/progress/comparisons.csv").open() as handle:
    assert len(list(csv.DictReader(handle))) == 30
families = json.loads((ROOT / "results/progress/comparisons.json").read_text())
assert set(families) == {m for m,_ in MODELS}
expected_tests = {(h, k) for h in ("H1", "H1b", "H2", "H3", "H4") for k in ("bertscore", "rougeL")}
assert all(len(f) == 10 and {(c["hypothesis"],c["metric"]) for c in f} == expected_tests
           and all(c["n_pairs"] == 1000 and 0 <= c["p_adjusted"] <= 1 for c in f)
           for f in families.values())
with (ROOT / "results/progress/scores.csv").open() as handle:
    scores = list(csv.DictReader(handle))
key = lambda r: (r["model"], r["doc_id"], r["arm"])
by_key = {key(r): r for r in scores}
assert len(scores) == len(by_key) == 18000
assert set(by_key) == {(m,d,a) for m,_ in MODELS for d in ids for a in ARMS}
with (ROOT / "results/audit520/scores-520.csv").open() as handle:
    old_scores = list(csv.DictReader(handle))
assert len(old_scores) == 9360 and all(by_key[key(r)] == r for r in old_scores)
for row in scores:
    for metric in ("bertscore", "bertscore_precision", "bertscore_recall", "rougeL", "meteor"):
        value = float(row[metric])
        assert math.isfinite(value) and 0 <= value <= 1
assert len(summary["token_audit"]) == 18000
assert all(not p["truncated"] and p["full_tokens"] == p["evaluated_tokens"]
           for pair in summary["token_audit"] for p in pair)
for path, sha in summary["method"].items():
    assert digest(ROOT / path) == sha
manifest_path = ROOT / "results/audit1000/raw-artifact-manifest.csv"
if manifest_path.exists():
    with manifest_path.open() as handle:
        complete_manifest = list(csv.DictReader(handle))
    assert len(complete_manifest) == 18000
    for row in complete_manifest:
        for part in ("request", "response"):
            assert digest(ROOT / row[f"{part}_path"]) == row[f"{part}_sha256"]
    print("All 18000 published raw hashes verified", flush=True)
report.update(
    scores=18000, original_scores_preserved=9360, original_raw_generations_preserved=9360,
    bertscore_full_coverage=True, method_id=summary["method_id"],
    progress_panel_hypotheses_tested=summary["hypotheses_tested"],
    confirmatory_tests_validated=30,
    reported_round_total_usd=str(sum(Decimal(m["reported_round_cost_usd"]) for m in report["models"].values())),
    held_round_total_usd=str(sum(Decimal(m["held_round_cost_usd"]) for m in report["models"].values())),
)
(OUT / "final-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
