"""Adendos historicos do Ling pago e teto incremental por rodada, sem API."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
OLD_SHA = "e7621072479ad9afec4e287e07c05d9d8a00adbb854526f1eba4a921e814d928"
NEW_PATH = ROOT / "configs/ling-paid-round-321-520.json"


@pytest.fixture
def paid(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("scripts.execution.run_prepared_paid")


@pytest.fixture
def rounds(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("scripts.execution.run_full_round")


@pytest.fixture
def core(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    module = importlib.import_module("scripts.execution.run_paid_concurrent")
    monkeypatch.setattr(module, "dispatch_delay", lambda *args: 0)
    return module


def new_sha():
    return hashlib.sha256(NEW_PATH.read_bytes()).hexdigest()


def frozen_cohort():
    return json.loads((ROOT / "configs/full_openrouter.lock.json").read_text())["cohort"]


class Clock:
    def __init__(self):
        self.now = 10000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def forbidden(*args, **kwargs):
    raise AssertionError("chamada de API proibida neste teste")


def test_registered_amendments_are_pinned_disjoint_and_keep_old_contract(paid):
    amendments = paid.paid_ling_amendments()
    assert [(a["first_case"], a["last_case_exclusive"]) for a, _ in amendments] == [
        (960, 1920),
        (1920, 3120),
    ]
    (old, old_sha), (new, sha) = amendments
    assert old_sha == OLD_SHA and sha == new_sha()
    assert paid.paid_ling_budgets(old) == (paid.money("2.00"), None)
    assert paid.paid_ling_budgets(new) == (paid.money("3.70"), paid.money("1.70"))
    assert sum(paid.money(v) for v in new["round_budget_usd"].values()) == paid.money("7.00")
    assert new["previous_amendments_sha256"] == [OLD_SHA]
    assert new["requested_cases"] == 200


def test_edited_or_unregistered_amendment_is_rejected(paid, tmp_path, monkeypatch):
    copy = tmp_path / "ling-paid-round-321-520.json"
    copy.write_bytes(NEW_PATH.read_bytes())
    with pytest.raises(ValueError, match="adendo"):
        paid.paid_ling_amendment(copy)
    edited = json.loads(NEW_PATH.read_text()) | {"reason": "edited later"}
    copy.write_text(json.dumps(edited))
    monkeypatch.setattr(paid, "PAID_LING_AMENDMENTS", ((copy, new_sha()),))
    with pytest.raises(ValueError, match="adendo"):
        paid.paid_ling_amendment(copy)
    # Mesmo registrado com o proprio hash, o contrato v2 exige tetos coerentes.
    broken = json.loads(NEW_PATH.read_text()) | {"round_budget_total_usd": "6.00"}
    copy.write_text(json.dumps(broken))
    digest = hashlib.sha256(copy.read_bytes()).hexdigest()
    monkeypatch.setattr(paid, "PAID_LING_AMENDMENTS", ((copy, digest),))
    with pytest.raises(ValueError, match="adendo"):
        paid.paid_ling_amendment(copy)


def test_saved_paid_attempt_needs_the_amendment_of_its_own_case(paid, tmp_path):
    target = paid.TARGETS["ling-paid-novita"]
    free = paid.TARGETS["ling-free"]
    row = {"doc_id": "test", "arm": "C1", "prompt_tokens": 100, "messages": []}
    for case in (960, 1920, 3119, 3120):
        (tmp_path / f"{case:04d}-0.request.json").write_text(
            json.dumps(paid.request_payload(target, row))
        )

    def entry(case, digest):
        return {
            "case": case,
            "response_file": f"{case:04d}-0.response.json",
            "amendment_sha256": digest,
        }

    for good in (entry(960, OLD_SHA), entry(1920, new_sha()), entry(3119, new_sha())):
        assert paid.target_for_saved_attempt(free, tmp_path, good, row) == target
    for bad in (entry(1920, OLD_SHA), entry(960, new_sha()), entry(3120, new_sha())):
        with pytest.raises(ValueError, match="adendo"):
            paid.target_for_saved_attempt(free, tmp_path, bad, row)


def test_new_amendment_matches_only_the_deterministic_321_520_plan(rounds, tmp_path):
    records = frozen_cohort()
    done = {model: set(range(1920)) for model in rounds.MODELS}
    plan = rounds.round_plan(tmp_path, records, done, 200, persist=False)
    assert plan["documents"] == [r["doc_id"] for r in records[320:520]]
    amendment, digest = rounds.round_amendment(plan, records)
    assert digest == new_sha() and amendment["plan_id"] == plan["plan_id"]
    assert rounds.paid_ling_for_plan(plan, records) == digest
    money = rounds.money
    assert rounds.amendment_round_budgets(amendment) == {
        "ling-free": money("1.70"),
        "qwen37": money("3.40"),
        "gemma26": money("1.90"),
    }
    for count in (100, 160, 201):
        other = rounds.round_plan(tmp_path, records, done, count, persist=False)
        assert rounds.round_amendment(other, records) is None
        with pytest.raises(ValueError, match="adendo"):
            rounds.paid_ling_for_plan(other, records)
    history = {model: set(range(960)) for model in rounds.MODELS}
    old_plan = rounds.round_plan(tmp_path, records, history, 160, persist=False)
    old, old_digest = rounds.round_amendment(old_plan, records)
    assert old_digest == OLD_SHA and rounds.amendment_round_budgets(old) == {}
    assert not (tmp_path / "active-round.json").exists()


def test_paid_flag_validates_round_before_saving_and_passes_caps(rounds, tmp_path, monkeypatch):
    records = frozen_cohort()
    accepted = {model: set(range(1920)) for model in rounds.MODELS}
    monkeypatch.setattr(rounds, "ROOT", tmp_path)
    monkeypatch.setattr(rounds, "verify_full_lock", lambda _: {"cohort": records})
    monkeypatch.setattr(rounds, "inspect_models", lambda _: ({"ling-free": None}, accepted))
    monkeypatch.setattr(rounds, "audit_batch", lambda *args: None)
    monkeypatch.setattr(rounds, "execute_round", forbidden)
    monkeypatch.setattr("sys.argv", ["round", "100", "--ling-paid-this-round"])
    with pytest.raises(ValueError, match="adendos"):
        rounds.main()
    assert not (tmp_path / "active-round.json").exists()

    monkeypatch.setattr("sys.argv", ["round", "200", "--dry-run", "--ling-paid-this-round"])
    rounds.main()
    preview = json.loads((tmp_path / "preview.json").read_text())
    assert preview["amendment_sha256"] == new_sha() and preview["ling_paid"]
    assert preview["round_budget_usd"] == {"ling-free": "1.70", "qwen37": "3.40", "gemma26": "1.90"}
    assert preview["pending_generations"] == {model: 1200 for model in rounds.MODELS}
    assert not (tmp_path / "active-round.json").exists()

    seen = {}

    def execute(plan, *args, **kwargs):
        seen.update(kwargs, plan=plan)
        return {"complete": True, "pending_generations": {}}

    monkeypatch.setattr(rounds, "execute_round", execute)
    monkeypatch.setattr(rounds, "update_metrics", lambda *args: None)
    monkeypatch.setattr("sys.argv", ["round", "200", "--ling-paid-this-round"])
    rounds.main()
    assert seen["paid_ling"] == new_sha()
    assert seen["round_budgets"]["qwen37"] == rounds.money("3.40")
    saved = json.loads((tmp_path / "active-round.json").read_text())
    assert saved == seen["plan"] and saved["plan_id"] == json.loads(NEW_PATH.read_text())["plan_id"]


@pytest.mark.parametrize("paid_ling", [None, "a" * 64])
def test_round_caps_reach_only_paid_executors(rounds, tmp_path, monkeypatch, paid_ling):
    records = [{"doc_id": str(i)} for i in range(1000)]
    accepted = {model: set() for model in rounds.MODELS}
    states = {
        m: SimpleNamespace(model=m, accepted=lambda m=m: accepted[m]) for m in rounds.MODELS
    }
    plan = rounds.round_plan(tmp_path, records, accepted, 10, persist=True)
    seen = {}

    def execute(state, *, documents, execute, stop_event=None, **kwargs):
        seen[state.model] = kwargs
        accepted[state.model].update(rounds.document_indices(records, documents))

    monkeypatch.setattr(rounds, "run_selection", execute)
    money = rounds.money
    caps = {"ling-free": money("1.7"), "qwen37": money("3.4"), "gemma26": money("1.9")}
    result = rounds.execute_round(
        plan, records, states, accepted, {}, tmp_path, paid_ling=paid_ling, round_budgets=caps
    )
    assert result["complete"]
    assert seen["qwen37"] == {"round_budget": caps["qwen37"]}
    assert seen["gemma26"] == {"round_budget": caps["gemma26"]}
    expected = {"paid_ling": paid_ling, "round_budget": caps["ling-free"]} if paid_ling else {}
    assert seen["ling-free"] == expected


def paid_state(tmp_path, monkeypatch, model="ling-free"):
    batcher = importlib.import_module("scripts.execution.run_ling_batches")
    adaptive = importlib.import_module("scripts.execution.run_paid_concurrent")
    records = [{"doc_id": str(i)} for i in range(1000)]
    rows = [{"doc_id": r["doc_id"], "arm": arm} for r in records for arm in batcher.ARMS]
    budget = batcher.money(0 if model == "ling-free" else 18)
    state = batcher.PreparedRun(model, tmp_path, tmp_path, records, rows, None, {}, budget)
    accepted = set()
    monkeypatch.setattr(batcher.PreparedRun, "accepted", lambda _: accepted.copy())
    monkeypatch.setattr(batcher, "ensure_meteor_resources", lambda: None)
    monkeypatch.setattr(batcher, "load_key", lambda: "mock")
    monkeypatch.setattr(
        batcher, "OpenRouterFreeClient", lambda *a, **k: SimpleNamespace(quota=forbidden)
    )
    monkeypatch.setattr(batcher, "run", forbidden)
    seen = {}

    def concurrent(rows, target, folder, budget, identity, transport, *, case_indices, **kwargs):
        seen.update(target=target, budget=budget, cases=set(case_indices), **kwargs)
        accepted.update(case_indices)

    monkeypatch.setattr(adaptive, "run", concurrent)
    return batcher, state, seen


def test_paid_ling_selection_uses_amendment_budgets_and_hash(paid, tmp_path, monkeypatch):
    batcher, state, seen = paid_state(tmp_path, monkeypatch)
    docs = [str(i) for i in range(320, 520)]
    batcher.run_selection(
        state, documents=docs, execute=True, paid_ling=new_sha(), round_budget=paid.money("5")
    )
    assert seen["target"] == paid.TARGETS["ling-paid-novita"]
    assert seen["budget"] == paid.money("3.70") and seen["round_budget"] == paid.money("1.70")
    assert seen["amendment_sha256"] == new_sha() and seen["cases"] == set(range(1920, 3120))
    with pytest.raises(ValueError, match="fora do adendo"):
        batcher.run_selection(
            state, documents=[str(i) for i in range(160, 320)], execute=True, paid_ling=new_sha()
        )
    with pytest.raises(ValueError, match="nao registrado"):
        batcher.run_selection(state, documents=docs, execute=True, paid_ling="b" * 64)
    seen.clear()
    batcher.run_selection(
        state, documents=[str(i) for i in range(160, 320)], execute=True, paid_ling=OLD_SHA
    )
    assert seen["budget"] == paid.money("2.00") and "round_budget" not in seen


def test_round_cap_is_rejected_for_free_ling_and_passed_to_paid_models(
    paid, tmp_path, monkeypatch
):
    batcher, state, _ = paid_state(tmp_path / "ling", monkeypatch)
    with pytest.raises(ValueError, match="teto incremental"):
        batcher.run_selection(
            state, documents=["0"], execute=True, round_budget=paid.money("1")
        )
    batcher, state, seen = paid_state(tmp_path / "qwen", monkeypatch, "qwen37")
    batcher.run_selection(
        state, documents=["0", "1"], execute=True, round_budget=paid.money("3.40")
    )
    assert seen["target"] == paid.TARGETS["qwen37"] and seen["budget"] == paid.money(18)
    assert seen["round_budget"] == paid.money("3.40") and "amendment_sha256" not in seen


def test_round_budget_ignores_other_cases_and_stops_before_overdraft(core, tmp_path):
    target = core.TARGETS["qwen37"]
    rows = [
        dict(
            doc_id=str(i),
            arm="C1",
            prompt_tokens=10,
            max_output_tokens=8192,
            messages=[{"role": "user", "content": str(i)}],
        )
        for i in range(3)
    ]
    reserve = core.reservation(target, rows[0])
    earlier = dict(
        case=0,
        doc_id="0",
        arm="C1",
        reserved_usd=str(reserve),
        status="accepted",
        at=0,
        response_file="0000-0.response.json",
        reported_cost_usd="0.5",
    )
    (tmp_path / "ledger.json").write_text(
        json.dumps(dict(identity={"same": True}, attempts=[earlier]))
    )
    calls = []

    def request(route, payload):
        calls.append(payload)
        return dict(
            model=target[0],
            provider=target[2],
            usage=dict(cost=0.001, prompt_tokens=10, completion_tokens=1),
            choices=[dict(finish_reason="stop", message=dict(content="summary"))],
        )

    clock = Clock()
    kwargs = dict(clock=clock, sleep=clock.sleep, jitter=lambda: 0)
    client = SimpleNamespace(_request=request)
    with pytest.raises(ValueError, match="casos da rodada"):
        core.run(rows, target, tmp_path, core.money(10), {"same": True}, client,
                 round_budget=core.money(1), **kwargs)
    with pytest.raises(ValueError, match="teto de gasto"):
        core.run(rows, target, tmp_path, core.money(10), {"same": True}, client,
                 case_indices={1, 2}, round_budget=reserve + core.money("0.0005"), **kwargs)
    assert len(calls) == 1
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert core.held_cost(ledger, {1, 2}) == core.money("0.001")
    assert core.held_cost(ledger) == core.money("0.501")
    # Sem o teto incremental, o teto cumulativo permite concluir sem repetir o caso aceito.
    core.run(rows, target, tmp_path, core.money(10), {"same": True}, client,
             case_indices={1, 2}, **kwargs)
    assert len(calls) == 2
    statuses = [a["status"] for a in json.loads((tmp_path / "ledger.json").read_text())["attempts"]]
    assert statuses == ["accepted", "accepted", "accepted"]
