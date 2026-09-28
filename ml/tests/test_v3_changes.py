"""Integrity gates for the v3 experiment (C1-C4, control, treatment).

These are the checks that must hold *before* a model is trained, because each one
invalidates the comparison if it fails silently:

  * the control must be byte-identical to v2, or it is not a control
  * no treatment text may appear in validation/test, or the scores are fiction
  * no target may be empty, or the model learns to emit nothing
  * C1 must remove boilerplate without eating content
  * C2 must never invent a verb, and must survive the v1 name redactor
  * the terse pool must be the size that was measured, not the size that was
    written down in a plan

Run: ``.venv/bin/python -m pytest ml/tests/test_v3_changes.py -q``
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from ml.cleaner import normalize_text
from ml.tune.build_dataset import DATA_DIR
from ml.tune.build_eval_v3 import BOILER_SENT
from ml.tune.v3_changes import (
    C1_SENTENCE_RE,
    TERSE_POOL_MIN_KERNEL,
    TERSE_UPSAMPLE,
    action_is_whole_sentence,
    clean_issue_c1,
    extract_action_c2,
    terse_pool,
)

ROOT = Path(__file__).resolve().parents[2]
TREATMENT = DATA_DIR / "tune" / "v3" / "treatment"
CONTROL = DATA_DIR / "tune" / "v3" / "control"
V2 = DATA_DIR / "tune" / "v2"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def _require(path: Path, builder: str):
    """Generated datasets are not all committed (see .gitignore). Rebuild, or
    fail with the command to run rather than an opaque FileNotFoundError."""
    if not (path.parent / "meta.json").exists():
        pytest.skip(f"run `{builder}` first: {path.parent} not built")
    return path


def _issues(chat: dict) -> list[dict]:
    text = next(m["content"] for m in chat["messages"] if m["role"] == "assistant")
    return json.loads(text)["topics"]


# --------------------------------------------------------------------- control
def test_control_train_is_byte_identical_to_v2():
    """The whole experiment rests on this: control == v2 data, seed only."""
    _require(CONTROL / "train.jsonl", "python -m ml.tune.build_control")
    assert (CONTROL / "train.jsonl").read_bytes() == (V2 / "train.jsonl").read_bytes()


def test_control_splits_are_byte_identical_to_v2():
    for split in ("validation", "test"):
        assert (CONTROL / f"{split}.jsonl").read_bytes() == (V2 / f"{split}.jsonl").read_bytes()


# -------------------------------------------------------------------- leakage
def test_no_treatment_text_leaks_into_held_out_splits():
    _require(TREATMENT / "train.jsonl", "python -m ml.tune.build_v3")
    """Keyed on normalised text, because uid is not an identity (audit s5)."""
    def texts(p: Path) -> set[str]:
        out = set()
        for chat in _load(p):
            for m in chat["messages"]:
                if m["role"] == "user":
                    out.add(normalize_text(m["content"]))
        return out

    train = texts(TREATMENT / "train.jsonl")
    for split in ("validation", "test"):
        held = texts(TREATMENT / f"{split}.jsonl")
        assert not (train & held), (
            f"{len(train & held)} texts shared between treatment train and {split}")


def test_treatment_held_out_splits_match_v2():
    """All three arms must be scored on identical held-out data."""
    for split in ("validation", "test"):
        assert (TREATMENT / f"{split}.jsonl").read_bytes() == (V2 / f"{split}.jsonl").read_bytes()


# --------------------------------------------------------------------- targets
def test_no_target_issue_is_empty():
    _require(TREATMENT / "train.jsonl", "python -m ml.tune.build_v3")
    """An empty issue is a target that teaches the model to emit nothing."""
    empty = [
        (chat.get("uid"), i)
        for chat in _load(TREATMENT / "train.jsonl")
        for i, t in enumerate(_issues(chat))
        if not t.get("issue", "").strip()
    ]
    assert not empty, f"{len(empty)} empty issues, e.g. {empty[:5]}"


def test_every_topic_matches_the_schema_shape():
    _require(TREATMENT / "train.jsonl", "python -m ml.tune.build_v3")
    allowed = {"domain", "issue", "object", "requested_action", "attributes"}
    for chat in _load(TREATMENT / "train.jsonl"):
        for t in _issues(chat):
            assert set(t) == allowed, f"{chat.get('uid')}: {sorted(t)}"


def test_domains_are_in_the_declared_enum():
    _require(TREATMENT / "train.jsonl", "python -m ml.tune.build_v3")
    from ml.tune.run_eval_v3 import SCHEMA_DOMAINS
    for chat in _load(TREATMENT / "train.jsonl"):
        for t in _issues(chat):
            assert t["domain"] in SCHEMA_DOMAINS, f"{t['domain']!r} not in enum"


# --------------------------------------------------------------------------- C1
def test_c1_removes_the_consent_sentence_whole():
    """Partial surgery on this sentence leaves wreckage, which is why C1 drops
    it at sentence level rather than by clause regex."""
    text = ("Прохання заасфальтувати двір. Заявник надає згоду на обробку своїх "
            "персональних даних та передачу їх третім особам відповідно до вимог ЗУ")
    out = clean_issue_c1(text)
    assert "персональних даних" not in out
    assert "заасфальтувати" in out
    assert not re.search(r"згоду\s+на\s+оброб", out, re.IGNORECASE)


def test_c1_removes_response_delivery():
    out = clean_issue_c1("Прохання розібратися. Відповідь надати поштою.")
    assert out == "Прохання розібратися"


def test_c1_removes_the_most_common_corpus_sentence():
    """Відповідь заявниці. occurs 1828 times and BOILER_SENT misses it: Ukrainian
    turns заявник's к into ц, so заявник\\w* cannot match заявниці."""
    assert clean_issue_c1("Відповідь заявниці. Вул. Козацька 3, яма.") == "Вул. Козацька 3, яма"
    assert clean_issue_c1("Відповідь заявнику.") == ""


def test_c1_keeps_meaningful_use_of_the_delivery_phrase():
    """'в телефонному режимі' is boilerplate in 'Відповідь в телефонному режимі'
    and content in a complaint about how staff spoke to the citizen."""
    out = clean_issue_c1("Скарга на працівницю, яка некоректно спілкувалася в телефонному режимі.")
    assert "телефонному режимі" in out


def test_c1_does_not_strip_real_content():
    for text, keep in [
        ("Яма на дорозі по вул. Шевченка 24. Прошу ремонт.", "Шевченка"),
        ("Про ремонт дороги та зелених зон.", "ремонт"),
        ("Усунути порив мережі водопостачання.", "порив"),
    ]:
        assert keep.lower() in clean_issue_c1(text).lower(), text


def test_c1_reduces_boilerplate_across_the_real_corpus():
    """The headline C1 number, measured rather than asserted on one string."""
    def rate(path: Path, pred) -> float:
        rows = [c for c in _load(path) if pred(c)]
        boiler = re.compile("|".join(BOILER_SENT), re.IGNORECASE)
        hits = sum(1 for c in rows for t in _issues(c)
                   if boiler.search(t.get("issue", "")))
        return hits / len(rows)

    single = lambda c: not str(c.get("uid", "")).startswith(("V3-TERSE", "MT-"))
    before = rate(DATA_DIR / "tune" / "train.jsonl", lambda c: True)
    after = rate(TREATMENT / "train.jsonl", single)
    assert before > 0.5, f"expected the v1 labels to be boilerplate-heavy, got {before}"
    assert after < before / 2, f"C1 only moved {before:.1%} -> {after:.1%}"


def test_c1_is_a_superset_of_the_v1_admin_clause_behaviour():
    """C1 must not reintroduce anything _ADMIN_CLAUSE already removed."""
    from ml.tune.build_dataset import _ADMIN_CLAUSE
    cases = [
        "Заявник повідомляє, що яма. Прошу ремонт.",
        "Скарга на працівника. Відповідь надати.",
        "Прохання надати відповідь.",
    ]
    for text in cases:
        v1 = _ADMIN_CLAUSE.sub("", text)
        v3 = clean_issue_c1(text)
        for token in ("Заявник повідомляє", "Відповідь надати", "Прохання надати"):
            if token not in v1:
                assert token not in v3, f"C1 reintroduced {token!r} in {text!r}"


# --------------------------------------------------------------------------- C2
def test_c2_returns_empty_when_the_request_names_no_action():
    assert extract_action_c2("Прохання вжити заходи реагування.") == ""
    assert extract_action_c2("Дощ іде, нічого не робиться.") == ""


def test_c2_keeps_the_verb_the_v1_redactor_would_eat():
    """_PERSON_NAME matches any capitalised word, so a sentence-initial
    imperative looks like a person's name and used to be redacted away."""
    action = extract_action_c2("Усунути порив мережі водопостачання, яке спричинило яму.")
    assert action.startswith("Усунути"), action
    assert "порив мережі" in action


def test_c2_extracts_a_clause_not_a_whole_sentence():
    text = "На вул. Шевченка яма. Прошу провести ямковий ремонт асфальтного покриття."
    action = extract_action_c2(text)
    assert action == "провести ямковий ремонт асфальтного покриття"
    assert not action_is_whole_sentence(action, text)


def test_c2_never_invents_a_verb():
    """No word in an action may be absent from the complaint it came from.

    Not a literal-substring test: ``_redact_pii`` legitimately rewrites the span
    ("...по пров. Ольги Кобилянської, 18" -> "...по пров. , 18"), so the action is
    not always a verbatim slice. The invariant that actually matters is that
    nothing is *invented*, which is what the word-set check pins down.
    """
    for chat in _load(DATA_DIR / "train.jsonl")[:400]:
        text = chat.get("content") or ""
        action = extract_action_c2(text)
        if not action:
            continue
        # Tokenise on word boundaries, not whitespace: the corpus writes
        # "пров.Ольги" with no space, so a whitespace split would hide "пров".
        src = {w.lower() for w in re.findall(r"[\w']+", text, re.UNICODE)}
        new = {w.lower() for w in re.findall(r"[\w']+", action, re.UNICODE)} - src
        assert not new, f"action introduces {new}: {action!r}"


def test_c2_shrinks_nonempty_actions_on_the_real_corpus():
    def share(path: Path) -> float:
        rows = _load(path)
        n = sum(len(_issues(c)) for c in rows)
        k = sum(1 for c in rows for t in _issues(c) if t.get("requested_action"))
        return k / n

    before = share(DATA_DIR / "tune" / "train.jsonl")
    after = share(TREATMENT / "train.jsonl")
    assert before > 0.4, f"v1 actions were expected to be common, got {before}"
    assert after < before * 0.6, f"C2 only moved {before:.1%} -> {after:.1%}"


# --------------------------------------------------------------------------- C3
def test_terse_pool_is_the_measured_size_not_the_planned_one():
    """plan.json says 178 and short_note_analysis.md says 928. Measured against
    the current _note_kernel the terse pool is 246. This pins the measurement so
    a future change to _note_kernel cannot silently resize the augmentation."""
    from ml.tune.build_multitopic import _note_kernel
    pool = terse_pool(_load(DATA_DIR / "train.jsonl"), _note_kernel)
    assert len(pool) == TERSE_POOL_MIN_KERNEL, (
        f"terse pool is {len(pool)}, expected {TERSE_POOL_MIN_KERNEL}; the plan's 178 "
        "and the report's 928 are both stale")


def test_terse_rows_are_the_pool_upsampled():
    meta = json.loads((TREATMENT / "meta.json").read_text())
    assert meta["terse_pool"]["n_records"] == TERSE_POOL_MIN_KERNEL
    assert meta["upsample_factors"]["terse"] == TERSE_UPSAMPLE
    # A few pool records are boilerplate-only, so C1 empties their issue and
    # they are skipped; n_units is the count that survived.
    n = sum(1 for c in _load(TREATMENT / "train.jsonl")
            if str(c.get("uid", "")).startswith("V3-TERSE"))
    assert meta["terse_pool"]["n_units"] <= TERSE_POOL_MIN_KERNEL
    assert n == meta["terse_pool"]["n_units"] * TERSE_UPSAMPLE, n


def test_terse_examples_are_short_and_show_the_kernel():
    terse = [c for c in _load(TREATMENT / "train.jsonl")
             if str(c.get("uid", "")).startswith("V3-TERSE")]
    assert terse
    for chat in terse:
        user = next(m["content"] for m in chat["messages"] if m["role"] == "user")
        assert 25 <= len(user) < 150, len(user)


# --------------------------------------------------------------------------- C4
def test_c4_raises_the_real_share_of_the_multitopic_slice():
    meta = json.loads((TREATMENT / "meta.json").read_text())
    v3_real = meta["slices"]["multitopic"]["real_row_share"]
    # v2 weighted the 22 real texts 12x against 3x for 957 synthetic = 8.4%.
    assert v3_real > 0.084, f"real share only rose to {v3_real}"
    assert meta["slices"]["multitopic"]["real_row_share"] == v3_real


def test_multitopic_slice_is_still_multi_topic():
    rows = [c for c in _load(TREATMENT / "train.jsonl")
            if str(c.get("uid", "")).startswith("MT-")]
    assert rows
    for chat in rows:
        assert len(_issues(chat)) >= 2, chat.get("uid")


def test_c4_does_not_claim_more_real_texts_than_exist():
    """The real pool is 22 hand-annotated records. Upsampling must not be
    reported as if it were more data."""
    unique = {str(c.get("uid", "")).split("#")[0] for c in _load(TREATMENT / "train.jsonl")
              if str(c.get("uid", "")).startswith("MT-REAL")}
    assert len(unique) == 22, f"expected the 22 hand-annotated real records, got {len(unique)}"


# ------------------------------------------------------- control recipe parity
def test_control_recipe_matches_the_v2_recipe():
    """The control must differ from v2 in the seed handling and nothing else.

    v2 predates run_config.json, so its recipe is recorded in adapter_config.json.
    Fields that only affect logging are excluded: steps_per_report changes how
    often a line is printed, not what is trained.
    """
    v2 = json.loads((V2.parent / "adapters" / "qwen3-8b-lora-v2"
                     / "adapter_config.json").read_text())
    ctl = json.loads(
        (DATA_DIR / "tune" / "adapters" / "qwen3-8b-lora-v3-control" / "run_config.json").read_text())
    ignore = {"adapter_path", "data", "steps_per_report", "project_name",
              "report_to", "clear_cache_threshold", "test_batches", "config"}
    for key in sorted(set(v2) & set(ctl)):
        if key in ignore:
            continue
        assert v2[key] == ctl[key], f"{key}: v2={v2[key]!r} control={ctl[key]!r}"


def test_control_recipe_covers_every_hyperparameter_that_matters():
    """Guard against a hyperparameter being silently absent from both files."""
    v2 = json.loads((V2.parent / "adapters" / "qwen3-8b-lora-v2"
                     / "adapter_config.json").read_text())
    for key in ("iters", "batch_size", "grad_accumulation_steps", "learning_rate",
                "max_seq_length", "num_layers", "lora_parameters", "seed",
                "grad_checkpoint", "mask_prompt", "optimizer", "model"):
        assert key in v2, f"v2 recipe is missing {key}; the parity check cannot see it"


# ------------------------------------------------------------------ fusion arms
def test_fuse_inherits_the_serving_contract_rather_than_re_deriving_it():
    """The failure this guards: mlx_lm.fuse copies the chat template from the
    base HF repo, which is how v2 ended up needing a hand-patched template that
    honours enable_thinking. Every fused arm must inherit v2's verified files."""
    from ml.tune.fuse_v3 import INHERITED, V2_FUSED
    assert "chat_template.jinja" in INHERITED
    assert (V2_FUSED / "chat_template.jinja").exists()
    tpl = (V2_FUSED / "chat_template.jinja").read_text()
    assert "enable_thinking" in tpl, "v2 template does not honour enable_thinking"


def test_fuse_refuses_to_run_without_the_verified_template():
    """Silently producing an arm with a different template is worse than failing.

    A real adapter directory is passed on purpose: the point is to reach the
    template check, not the "is this an adapter" check in front of it.
    """
    from ml.tune import fuse_v3
    adapter = DATA_DIR / "tune" / "adapters" / "qwen3-8b-lora-v2"
    assert (adapter / "adapter_config.json").exists(), "need a real adapter to test with"
    with pytest.raises(SystemExit, match="serving contract"):
        fuse_v3.fuse(adapter, adapter.parent / "unused-out",
                     template_from=ROOT / "ml/tune" / "does-not-exist")


def test_fuse_rejects_a_directory_that_is_not_an_adapter():
    from ml.tune import fuse_v3
    with pytest.raises(SystemExit, match="not a LoRA adapter"):
        fuse_v3.fuse(fuse_v3.V2_FUSED, fuse_v3.DATA / "unused-out")


@pytest.mark.parametrize("arm", ["v3-control", "v3-treatment"])
def test_fused_arm_is_served_identically_to_v2(arm):
    """Skipped until the arm is trained; when it runs, a v3 arm must differ
    from the v2 baseline in weights and in nothing else."""
    from ml.tune.fuse_v3 import INHERITED, V2_FUSED
    import hashlib
    fused = DATA_DIR / "tune" / "adapters" / f"qwen3-8b-lora-{arm}-fused"
    if not (fused / "fuse_manifest.json").exists():
        pytest.skip(f"{arm} not fused yet")
    for name in INHERITED:
        a = hashlib.sha256((fused / name).read_bytes()).hexdigest()
        b = hashlib.sha256((V2_FUSED / name).read_bytes()).hexdigest()
        assert a == b, f"{name} differs between {arm} and v2"
    assert (fused / "model.safetensors").exists() or (fused / "model.safetensors.index.json").exists()
    assert json.loads((fused / "fuse_manifest.json").read_text())["fused_modules"] > 0


# ------------------------------------------------------------------ comparison
def test_mcnemar_is_one_when_the_arms_agree_everywhere():
    from ml.tune.compare_arms import mcnemar
    v = [True, False, True, False]
    r = mcnemar(v, list(v))
    assert r["a_only"] == 0 and r["b_only"] == 0 and r["p"] == 1.0


def test_mcnemar_uses_only_the_discordant_pairs():
    """N agreements must not dilute the test: 100 shared successes say nothing
    about a difference concentrated in 10 cases."""
    from ml.tune.compare_arms import mcnemar
    a = [True] * 10 + [True] * 100
    b = [True] * 10 + [False] * 100
    r = mcnemar(a, b)
    assert (r["a_only"], r["b_only"]) == (100, 0)
    assert r["p"] < 0.01, r


def test_mcnemar_reports_no_difference_as_uncertain():
    from ml.tune.compare_arms import mcnemar
    a = [True] * 6 + [False] * 94
    b = [True] * 5 + [False] * 95
    assert mcnemar(a, b)["p"] > 0.05


def test_bootstrap_ci_brackets_the_observed_delta():
    from ml.tune.compare_arms import paired_bootstrap
    a = [1.0] * 50 + [0.0] * 50
    b = [0.0] * 50 + [0.0] * 50
    r = paired_bootstrap(a, b, iters=2000)
    assert r["delta"] == 0.5
    lo, hi = r["ci95"]
    assert lo <= 0.5 <= hi


def test_taxonomy_attributes_each_failure_once():
    """First match wins, so a case that is not valid JSON is not also counted as
    a domain error -- otherwise one broken case inflates several categories."""
    from ml.tune.compare_arms import classify
    case = {"json_ok": False, "schema_ok": False, "topic_count_ok": False,
            "domain_set_exact": False, "domain_set_recall": 0.0,
            "domain_set_precision": 0.0, "boilerplate_leak": True,
            "action_redundant": True, "out_of_enum_domain": True,
            "duplicate_domain": True}
    assert classify(case) == "not_json"


def test_taxonomy_counts_every_case_exactly_once():
    """no_error + failures == n, so no case is dropped or double-counted."""
    from ml.tune.compare_arms import classify
    for tag in ("v2", "floor_reference"):
        p = ROOT / "ml/data/tune/eval_v3/results" / f"{tag}.json"
        if not p.exists():
            pytest.skip(f"{tag} results not present")
        result = json.loads(p.read_text())
        labels = [classify(c) for c in result["per_case"]]
        assert len(labels) == result["metrics"]["n"]
        assert sum(1 for x in labels if x is None) + sum(
            1 for x in labels if x is not None) == len(labels)


def test_reference_floor_reproduces_its_own_references():
    """If the floor cannot score 1.0 against labels built from itself, the scorer
    is wrong and every model number measured with it is wrong too."""
    p = ROOT / "ml/data/tune/eval_v3/results/floor_reference.json"
    if not p.exists():
        pytest.skip("floor results not present")
    m = json.loads(p.read_text())["metrics"]
    assert m["issue_rouge_l_best"] == 1.0, m
    assert m["domain_set_precision"] == 1.0, m
    assert m["domain_set_recall"] == 1.0, m
    assert m["topic_count_accuracy"] == 1.0, m


def test_compare_refuses_two_different_case_suites():
    """A per-case diff silently drops unshared cases; the number it prints would
    then describe a subset neither arm was measured on."""
    from ml.tune import compare_arms
    orig = compare_arms.load
    try:
        def fake(tag):
            r = orig(tag)
            r["per_case"] = r["per_case"][:-1] if tag == "floor_reference" else r["per_case"]
            return r
        compare_arms.load = fake
        with pytest.raises(SystemExit, match="same cases"):
            compare_arms.compare("v2", "floor_reference")
    finally:
        compare_arms.load = orig
