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
    # the "у" euphony form is the same phrase and must be kept here too
    out = clean_issue_c1("Скарга на працівницю, яка некоректно спілкувалася у телефонному режимі.")
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


# ------------------------------------------------------------------- R2 strand
def test_r2_removes_the_phone_mode_delivery_strand():
    """R2: "Відповідь заявниці в телефонному режимі." survived C1's response
    sentence rule (заявниці intervenes) and its clause rule then stranded the
    bare tail. A stranded delivery-mode tail names no complaint, so it is
    removed whole -- including the dative-recipient form the multitopic composer
    leaves behind ("надати заявниці в телефонному режимі").

    Both euphony forms are covered: the corpus writes "у телефонному режимі"
    127 times against 584 "в", so a ``в``-only rule silently leaves a fifth of
    the records behind."""
    assert clean_issue_c1("Відповідь заявниці в телефонному режимі.") == ""
    assert clean_issue_c1("Відповідь заявниці у телефонному режимі.") == ""
    # the dominant surviving shape: the subject sits between "Відповідь" and the
    # verb, so the response-delivery rule cannot match it
    assert clean_issue_c1("Відповідь заявниця бажає отримати у телефонному режимі.") == ""
    assert clean_issue_c1("Відповідь заявник бажає отримати в телефонному режимі.") == ""
    assert clean_issue_c1("Заявниця бажає отримати відповідь в телефонному режимі.") == ""
    assert clean_issue_c1("Надати заявниці в телефонному режимі.") == ""
    assert clean_issue_c1("Яма на дорозі. Надати заявнику в телефонному режимі.") \
        == "Яма на дорозі"
    assert clean_issue_c1("Яма на дорозі. Надати заявнику у телефонному режимі.") \
        == "Яма на дорозі"
    # a glued period+capital boundary is not split, so the stranded tail is
    # peeled by the boundary-anchored clause rule instead
    assert clean_issue_c1("Яма на дорозі.Відповідь заявниці у телефонному режимі.") \
        == "Яма на дорозі"
    # the delivery verb + recipient + mode clause, inside a retained sentence
    assert clean_issue_c1("Прошу надати заявниці в телефонному режимі інформацію про борг.") \
        == "Прошу інформацію про борг"


def test_r2_treatment_targets_carry_no_phone_mode_delivery_strand():
    """Corpus-level check of the R2 fix. The phone-mode phrase may still appear
    where it is content ("некоректно спілкувалася ... в телефонному режимі",
    "місце уточнити із заявницею в телефонному режимі"); what must be gone is
    the stranded delivery tail: a recipient noun in the dative immediately before
    it, or a bare "в/у телефонному режимі" left on its own. ``[ву]`` because the
    phrase is euphony-alternating."""
    _require(TREATMENT / "train.jsonl", "python -m ml.tune.build_v3")
    stranded = re.compile(
        r"(?:заявни(?:ку|ці|кам|цям)\s+)?[ву]\s+у?\s*телефонному\s+режимі",
        re.IGNORECASE)
    # Meaningful use: the phone-mode phrase describes *how staff behaved* or
    # *how a detail is to be clarified*, introduced by an instrumental
    # "з заявником/із заявницею" (optionally with a comma before the phrase).
    instrumental = re.compile(
        r"(?:із|з)\s+заявни(?:цею|ком)\s*,?\s*[ву]\s+у?\s*телефонному\s+режимі",
        re.IGNORECASE)
    bad = []
    for chat in _load(TREATMENT / "train.jsonl"):
        for t in _issues(chat):
            issue = t.get("issue", "")
            if instrumental.search(issue):
                continue                      # meaningful, kept deliberately
            if stranded.search(issue):
                bad.append(issue[:110])
    assert not bad, f"{len(bad)} phone-mode delivery strands, e.g. {bad[:3]}"


# ----------------------------------------------------------------- R3 closer
def test_r3_generic_closer_is_end_anchored_not_partial():
    """R3: the closer rule must fire only when the sentence *is* the closer.
    "…вжити заходи та усунути причину витоку води." carries a concrete object
    after the generic head and was dropped whole before the end-anchor."""
    assert clean_issue_c1("Прохання вжити заходи реагування.") == ""
    assert clean_issue_c1("Двір засмічений. Прохання вжити заходи реагування.") \
        == "Двір засмічений"
    assert "усунути причину" in clean_issue_c1(
        "Прохання вжити заходи та усунути причину витоку води.")
    assert "для відновлення" in clean_issue_c1(
        "Прохання вжити заходи для відновлення води.")


def test_r3_clause_guard_leaves_the_purpose_after_the_closer():
    """The clause rule drops the closer only at a clause boundary, so a purpose
    clause introduced by "для"/"щодо" is never partially eaten."""
    out = clean_issue_c1("Прохання вжити заходів щодо усунення трупного запаху "
                         "у квартирі №30.")
    assert "усунення трупного запаху" in out
    action = extract_action_c2("Прохання вжити заходи для відновлення води.")
    assert action == "вжити заходи для відновлення води"



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


def test_r1_terse_domains_are_derived_from_kind_not_stamped_other():
    """R1: _terse_examples hardcoded domain="other" for every terse row, so the
    whole C3 augmentation taught the model one label. The pool records carry
    ``kind``; derive the domain from it like every other stream does. All 234
    units map to real domains -- none is left as "other"."""
    from ml.tune.build_multitopic import _note_kernel
    from ml.tune.build_dataset import _clean_text, KIND_TO_DOMAIN

    pool = terse_pool(_load(DATA_DIR / "train.jsonl"), _note_kernel)
    domains = [
        KIND_TO_DOMAIN.get(r.get("kind") or "", "other")
        for r in pool
        if clean_issue_c1(_clean_text(_note_kernel(r.get("content") or ""))).strip()
    ]
    assert domains, "terse pool produced no usable units"
    assert "other" not in domains, (
        f"{sum(d == 'other' for d in domains)} terse units still stamped 'other'")
    # and the built dataset shows the same real spread
    seen = {t["domain"] for c in _load(TREATMENT / "train.jsonl")
            if str(c.get("uid", "")).startswith("V3-TERSE") for t in _issues(c)}
    assert seen and "other" not in seen, sorted(seen)


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


# ----------------------------------------------------------------------- gates
def test_gates_are_not_vacuous_v2_must_fail_them():
    """A gate that every existing artifact passes is not a gate. v2 is the
    reference for every number, so if v2 passes, the target is set wrong."""
    from ml.tune.gates_v3 import evaluate
    rep = evaluate("v2")
    assert rep.verdict() == "FAIL", [g.name for g in rep.blocking_failed]
    failed = {g.name for g in rep.blocking_failed}
    assert {"schema", "boilerplate", "topic_count", "smoke_two_topic"} <= failed


def test_all_nine_gates_are_evaluated():
    from ml.tune.gates_v3 import evaluate
    rep = evaluate("v2")
    assert len(rep.gates) == 9, [g.name for g in rep.gates]
    names = [g.name for g in rep.gates]
    assert len(set(names)) == 9, names
    assert {g.name for g in rep.gates if g.blocking} == {
        "schema", "boilerplate", "topic_count", "two_topic_recall",
        "frozen_domain", "smoke_two_topic"}


def test_every_gate_states_the_suite_it_was_measured_on():
    """The 329 and 85 suites score against v1 weak labels that still contain the
    boilerplate C1 removes, so a gate that does not name its suite cannot be read."""
    from ml.tune.gates_v3 import evaluate
    for g in evaluate("v2").gates:
        assert g.suite and g.target and g.baseline


def test_gate_observed_values_match_the_frozen_record():
    """The evaluator must read the recorded numbers, not re-derive or hardcode."""
    from ml.tune.gates_v3 import evaluate
    by_name = {g.name: g for g in evaluate("v2").gates}
    ev3 = json.loads((ROOT / "ml/data/tune/eval_v3/results/v2.json").read_text())["metrics"]
    assert by_name["boilerplate"].observed == ev3["boilerplate_leak_rate"]
    assert by_name["topic_count"].observed == ev3["topic_count_accuracy"]
    frozen = json.loads((ROOT / "ml/data/tune/eval/lora.json").read_text())["metrics"]
    assert by_name["frozen_domain"].observed == frozen["domain_accuracy"]
    mt = json.loads((ROOT / "ml/data/tune/multitopic/results/lora.json").read_text())["metrics"]
    assert by_name["two_topic_recall"].observed == mt["multi_topic_recall"]


def test_smoke_two_topic_gate_counts_only_multitopic_cases():
    from ml.tune.gates_v3 import _smoke_two_topic, _smoke
    rows, path = _smoke("v2")
    assert rows, path
    ok, total = _smoke_two_topic(rows)
    mt = [r for r in rows if r["category"] == "Multi-Topic"]
    assert total == len(mt) == 4, total
    assert 0 <= ok <= total


def test_empty_action_gate_counts_invented_actions_in_category_h():
    from ml.tune.gates_v3 import _empty_action_invention, _eval_v3
    ev3 = _eval_v3("v2")
    h = [c for c in ev3["per_case"] if c["category"] == "H"]
    assert len(h) == 12, len(h)
    expected = sum(1 for c in h if any((a or "").strip() for a in c["predicted_actions"]))
    assert _empty_action_invention(ev3) == expected


# ------------------------------------------------------------- training completion
def _fake_adapters(tmp_path: Path, iters_done: int, save_every: int = 100,
                   final_marker: bool = True) -> tuple[Path, Path]:
    d = tmp_path / "adapters"
    d.mkdir()
    for it in range(save_every, iters_done + 1, save_every):
        (d / f"{it:07d}_adapters.safetensors").write_bytes(b"x")
    # the file mlx_lm rewrites at every checkpoint
    (d / "adapters.safetensors").write_bytes(b"x")
    log = tmp_path / "train.log"
    log.write_text("Iter 100: Saved adapter weights\n"
                   + ("Saved final weights to adapters.safetensors.\n"
                      if final_marker else ""))
    return d, log


def test_presence_of_adapters_safetensors_is_not_completion(tmp_path):
    """The exact trap. mlx_lm writes adapters.safetensors at *every* checkpoint,
    so a pipeline waiting on that file fuses a 100-iteration adapter and reports
    it as the finished arm -- same size, same shape, silently wrong."""
    from ml.tune.training_state import is_complete
    d, log = _fake_adapters(tmp_path, iters_done=100, final_marker=False)
    assert (d / "adapters.safetensors").exists()
    assert not is_complete(d, 800, 100, log), "an unfinished run reported complete"


def test_complete_requires_every_checkpoint_and_the_final_marker(tmp_path):
    from ml.tune.training_state import is_complete
    d, log = _fake_adapters(tmp_path, iters_done=800, final_marker=True)
    assert is_complete(d, 800, 100, log)


def test_missing_final_marker_is_not_complete(tmp_path):
    """All eight checkpoints on disk but the loop never finished: a crash between
    the last periodic save and the end of training."""
    from ml.tune.training_state import is_complete
    d, log = _fake_adapters(tmp_path, iters_done=800, final_marker=False)
    assert not is_complete(d, 800, 100, log)


def test_a_gap_in_the_checkpoint_sequence_is_not_complete(tmp_path):
    from ml.tune.training_state import is_complete
    d, log = _fake_adapters(tmp_path, iters_done=800, final_marker=True)
    (d / "0000400_adapters.safetensors").unlink()
    assert not is_complete(d, 800, 100, log)


def test_describe_reports_what_is_still_missing(tmp_path):
    from ml.tune.training_state import describe
    d, _ = _fake_adapters(tmp_path, iters_done=300)
    info = describe(d, 800, 100)
    assert info["checkpoints"] == [100, 200, 300]
    assert info["missing"] == [400, 500, 600, 700, 800]
    assert info["n_checkpoints"] == 3


def test_the_live_control_run_is_reported_incomplete():
    """Guards the run in flight: if this ever passes while the job is still
    going, the driver is about to fuse a partial adapter."""
    from ml.tune.training_state import describe
    d = DATA_DIR / "tune" / "adapters" / "qwen3-8b-lora-v3-control"
    if not d.exists():
        pytest.skip("control run not started")
    info = describe(d, 800, 100)
    assert not info["final_checkpoint_present"] or info["n_checkpoints"] == 8


# ------------------------------------------------- C1/C2 on the multitopic slice
def _topics_of(chat: dict) -> list[dict]:
    return json.loads(next(m["content"] for m in chat["messages"]
                           if m["role"] == "assistant"))["topics"]


def _multitopic_chat(issues: list[str]) -> dict:
    return {
        "uid": "MT-TEST-1",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": " ".join(issues) + " Відповідь надати поштою."},
            {"role": "assistant", "content": json.dumps(
                {"topics": [{"domain": "roads", "issue": i, "object": "",
                             "requested_action": "", "attributes": {}}
                            for i in issues]}, ensure_ascii=False)},
        ],
    }


def test_multitopic_relabel_keeps_topics_distinct():
    """The regression this guards. _relabel derives every topic from the single
    merged user text, so reusing it here would give every topic the same merged
    issue and erase the per-topic distinction the multitopic suite exists for."""
    from ml.tune.build_v3 import _relabel_multitopic
    chat = _multitopic_chat(["яма на дорозі по вул. Шевченка",
                             "сміття не вивозять"])
    out, _ = _relabel_multitopic(chat)
    issues = [t["issue"] for t in _topics_of(out)]
    assert len(set(issues)) == 2, issues
    assert "Шевченка" in issues[0] and "сміття" in issues[1]


def test_multitopic_relabel_cleans_each_topic_independently():
    from ml.tune.build_v3 import _relabel_multitopic
    chat = _multitopic_chat([
        "ситуація по вул. Жадова 19. Заявник надає згоду на обробку персональних даних",
        "не вивозять сміття",
    ])
    out, before = _relabel_multitopic(chat)
    issues = [t["issue"] for t in _topics_of(out)]
    assert not any("персональних даних" in i for i in issues), issues
    assert "Жадова" in issues[0]
    assert "сміття" in issues[1], "the clean topic must not be touched"
    assert before["boilerplate_issue"] == 1


def test_every_training_slice_is_clean_of_boilerplate():
    """C1 has to reach all three slices. It originally reached one, which would
    have trained the model to emit consent text on 26% of the rows."""
    boiler = re.compile("|".join(BOILER_SENT), re.IGNORECASE)
    slices = {
        "single": lambda c: not str(c.get("uid", "")).startswith(("V3-TERSE", "MT-")),
        "terse": lambda c: str(c.get("uid", "")).startswith("V3-TERSE"),
        "multitopic": lambda c: str(c.get("uid", "")).startswith("MT-"),
    }
    rows = _load(TREATMENT / "train.jsonl")
    for name, pred in slices.items():
        topics = [t for c in rows if pred(c) for t in _issues(c)]
        rate = sum(1 for t in topics if boiler.search(t.get("issue", ""))) / len(topics)
        assert rate < 0.10, f"{name} slice still {rate:.1%} boilerplate"


def test_no_multitopic_chat_lost_a_topic():
    """Dropping a boilerplate-only topic would silently turn a two-topic example
    into a one-topic one and corrupt the suite C3/C4 are measured on."""
    for chat in _load(TREATMENT / "train.jsonl"):
        if str(chat.get("uid", "")).startswith("MT-"):
            assert len(_issues(chat)) == 2, f"{chat['uid']} has {len(_issues(chat))} topics"


def test_the_drop_counts_are_recorded():
    """A silent data loss reads as a bug later; the number has to be in meta."""
    meta = json.loads((TREATMENT / "meta.json").read_text())
    dropped = meta["slices"]["dropped_empty_issue"]
    assert dropped["n_single"] > 0 and dropped["n_multitopic"] > 0, dropped
    assert "empty issue" in dropped["reason"]


def test_fuse_rel_handles_relative_and_absolute_paths():
    """Path.relative_to raises on a relative argument against an absolute root;
    fusion records provenance in the manifest, so this surfaced as a crash at
    the fuse step of the driver."""
    from ml.tune.fuse_v3 import _rel
    import os
    rel = "ml/data/tune/adapters/qwen3-8b-lora-v3-control"
    assert _rel(Path(rel)) == rel
    assert _rel(Path(rel).resolve()) == rel
    assert _rel(Path(rel)) == _rel(Path(rel).resolve())
    assert _rel(Path(os.getcwd())) == "."


def test_training_budget_covers_only_a_window_of_the_corpus():
    """The control arm failed its own validation for a mechanical reason: an
    800-step run at batch 2 consumes a random 18.8% of the 8519-example corpus,
    and evaluation lands on the ~81% of content the model never saw. This test
    pins the budget to a sub-epoch window so a future change to iters/batch that
    silently reintroduces (or fixes) the lottery is noticed. The driver refused
    to train the treatment because of this; do not raise iters/batch without
    re-reading ml/reports/v3_experiment_design.md §11 and re-validating the
    control."""
    n_examples = 8519
    budget_examples = 800 * 2        # args.iters x args.batch_size in the v2 recipe
    n_batches = (n_examples - 2) // 2 + 1
    coverage = budget_examples / n_examples
    assert n_batches > 800            # not even one epoch exists in 800 steps
    assert 0.15 < coverage < 0.25      # window lottery holds; update if protocol changes
    assert budget_examples < n_examples  # a run never sees the whole corpus


def test_the_control_did_not_pass_its_own_validation():
    """Freeze the negative result that stopped the experiment. The control is
    byte-identical otherwise to v2, and the head-to-head gap is the measured
    evidence that n=1 gating against v2's baselines is untrustworthy. If this
    ever needs to flip, the protocol's precondition must be re-derived first."""
    import json as _json
    v2 = _json.load(open(ROOT / "ml/data/tune/eval_v3/results/v2.json"))["metrics"]
    ctl = _json.load(open(ROOT / "ml/data/tune/eval_v3/results/v3-control.json"))["metrics"]
    gap = ctl["topic_count_accuracy"] - v2["topic_count_accuracy"]
    assert gap < -0.05, (
        f"control within {gap:+.4f} of v2 on topic_count; either the batch "
        "window lottery is gone or v3-control.json was overwritten")


def test_run_train_accepts_resume_flag():
    """--resume-adapter-file is what makes the full-corpus run resumable after
    an infrastructure crash; it must round-trip into the namespace."""
    from ml.tune.run_train import build_parser
    args = build_parser().parse_args(
        ["--adapter-path", "x", "--iters", "4724",
         "--resume-adapter-file", "ml/data/tune/adapters/qwen3-8b-lora-v3-treatment/04500_adapters.safetensors"])
    assert str(args.resume_adapter_file).endswith("04500_adapters.safetensors")


def test_full_corpus_iterations_cover_every_training_row():
    """The v3 replacement for the 800-step window: iters must be >= ceil(rows /
    batch) so every emitted training example is seen at least once."""
    from pathlib import Path
    rows = sum(1 for _ in open(TREATMENT / "train.jsonl"))
    batch = 2
    iters = (rows + batch - 1) // batch
    n_batches = (rows - batch) // batch + 1
    # Pinned recipe value: the run scripts pass --iters 4724 for this dataset.
    # The row count moved with the corrected R2/R3 (9 delivery-only single rows
    # removed, 12 closer-bearing rows rescued, 4 delivery-only multitopic rows
    # dropped), and ceil(9447/2) is 4724.
    assert rows == 9447
    assert iters == 4724
    assert iters >= n_batches          # not (as before) a random ~19% window
    assert iters * batch >= rows       # every row reachable in one epoch


def test_gates_smoke_reader_accepts_tag_dir(tmp_path, monkeypatch):
    """run_smoke --tag writes smoke/results/<tag>/; the gate table must read
    that layout (legacy finetuned-* was only the original v2 run)."""
    import ml.tune.gates_v3 as g
    d = tmp_path / "smoke" / "results"
    (d / "v3-treatment").mkdir(parents=True)
    (d / "v3-treatment" / "smoke_results.jsonl").write_text(
        '{"id":"smoke-13","category":"Multi-Topic","parsed":{"topics":[{"domains":[]},{"domains":[]}]}}\n'
        '{"id":"smoke-16","category":"Multi-Topic","parsed":{"topics":[{"domains":[]}]}}\n',
        encoding="utf-8")
    monkeypatch.setattr(g, "SMOKE", d)
    rows, path = g._smoke("v3-treatment")
    assert Path(path).name == "smoke_results.jsonl"
    assert Path(path).parent.name == "v3-treatment"
    assert g._smoke_two_topic(rows) == (1, 2)
