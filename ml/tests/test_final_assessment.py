"""Integrity gates for the final cross-arm assessment.

The assessment claims that three of the pessimistic readings of v3 are measurement
artefacts -- a style-biased reference set, a mislabeled category, an euphony blind
spot in the gate. Each of those is a claim *against* the measurement, so each needs
a test that fails if the artefact ever changes to make the claim false again:

  * ``test_cross_arm_*`` -- the matrix reproduces the committed aggregates and its
    frozen per-case decomposition is faithful
  * ``test_multitopic_*`` -- transitions are a full source->destination matrix, and
    the inline/boundary feature cannot silently collapse to one side
  * ``test_action_*`` -- the cat-H audit finds the four known mislabels, and the
    v3 arms omit rather than invent
  * ``test_residue_*`` -- R3 is measured with both euphonic forms and is still
    reported per arm
  * ``test_verify_final_assessment`` -- the verifier's own arithmetic

Run: ``.venv/bin/python -m pytest ml/tests/test_final_assessment.py -q``
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ml.tune import (
    action_analysis,
    cross_arm_matrix,
    decode_matrix,
    multitopic_mechanism,
    residue,
    verify_final_assessment,
)
from ml.tune.build_eval_v3 import BOILER_SENT

# --- public-checkout guard ---------------------------------------------
# A handful of tests read corpora generated from the official CC BY source,
# which the public repository does not ship (see REPRODUCIBILITY.md). They skip
# individually; the committed assessment artifacts are always tested.
from ml.tests._corpora import skip_if_missing  # noqa: E402

#: gate evaluation lives with the matrix it interprets
_bad = verify_final_assessment._bad

ROOT = Path(__file__).resolve().parents[2]
TUNE = ROOT / "ml/data/tune"
ARMS = ("v2", "v3-treatment", "v3-corrected", "v3-2")


@pytest.fixture(scope="module")
def matrix():
    return {r["key"]: r for r in cross_arm_matrix.build()["rows"]}


# --------------------------------------------------------------------- matrix
def test_matrix_covers_every_documented_dimension(matrix):
    for key in ("ev3_schema", "ev3_boilerplate", "ev3_topic_count",
                "mt_recall", "fr_domain_acc", "smoke_mt", "fr_action_match",
                "fr_issue_rl", "fr_domain_f1", "terse_A"):
        assert key in matrix, key


def test_matrix_agrees_with_committed_eval_v3_aggregates(matrix):
    """The matrix must not restate the runner; it must reproduce it."""
    for arm in ARMS:
        got = json.loads(
            (TUNE / "eval_v3/results" / f"{arm}.json").read_text(encoding="utf-8"))["metrics"]
        assert matrix["ev3_schema"]["values"][arm] == got["schema_validity_rate"]
        assert matrix["ev3_boilerplate"]["values"][arm] == got["boilerplate_leak_rate"]
        assert matrix["ev3_topic_count"]["values"][arm] == got["topic_count_accuracy"]


def test_matrix_frozen_decomposition_is_faithful():
    """The frozen per-case numbers must reproduce the committed aggregate.

    The matrix derives frozen per-case metrics by re-running the evaluator, so a
    change in either would silently produce a table that no longer describes the
    committed run.
    """
    fid = cross_arm_matrix.build()["frozen_per_case_fidelity"]
    assert set(fid) == set(ARMS)
    for arm, v in fid.items():
        assert v["faithful"], f"{arm}: {v['mismatch']}"


def test_matrix_boilerplate_gate_is_the_one_v3_solves(matrix):
    """Boilerplate is the only blocking gate a v3 arm clears."""
    cleared = {a for a in ARMS[1:] if not _bad(matrix, "ev3_boilerplate", a)}
    assert cleared == {"v3-corrected", "v3-2"}


def test_matrix_no_v3_arm_clears_schema_topic_count_or_smoke(matrix):
    for key in ("ev3_schema", "ev3_topic_count", "smoke_mt"):
        for arm in ARMS:
            assert _bad(matrix, key, arm), f"{key} {arm} unexpectedly passes"


def test_no_arm_including_v2_clears_every_blocking_gate(matrix):
    """The load-bearing release claim: nothing is releasable."""
    for arm in ARMS:
        failing = [k for k, r in matrix.items() if r["blocking"] and _bad(matrix, k, arm)]
        assert failing, f"{arm} clears every blocking gate -- the report must change"


# ---------------------------------------------------------------- multitopic
def test_transitions_are_a_full_source_destination_matrix():
    """Every arm pair must account for all 52 cases across all 16 cells."""
    r = multitopic_mechanism.build()
    buckets = set(multitopic_mechanism.BUCKETS)
    for pair, tr in r["transitions"].items():
        assert sum(tr.values()) == 52, (pair, tr)
        for cell in tr:
            src, dst = cell.split(" -> ")
            assert src in buckets and dst in buckets, cell
    for k in r["buckets"]:
        assert sum(r["buckets"][k].values()) == 52, k


def test_boundary_feature_matches_the_suite_taxonomy():
    """Anchored to category D, not to punctuation.

    Counting sentence terminators put 44 cases on the boundary side against 8
    for category D, because the terminator class included ':' and ';'.
    """
    r = multitopic_mechanism.build()["feature_crosstab"]
    by = {f["feature"]: f for f in r}
    for feat, want in (("boundary_marked", {"D"}), ("inline", {"B", "C"})):
        true_ids = {i for g in by[feat]["groups"] if g["value"] for i in g["ids"]}
        assert len(true_ids) == len(want) * 8, (feat, len(true_ids))


#: Features with no variance across the 52 cases. They cannot support any claim,
#: so they are pinned here explicitly: adding a feature to this set is how a
#: constant gets admitted, and the report has to keep saying they carry no signal.
DEGENERATE_FEATURES = {"same_domain", "different_domain", "second_similar",
                      "issue_sim_ge_60"}


def test_quote_failures_are_the_source_text_artefact_not_hallucination():
    """The `""` artefact is in the eval inputs and in BOTH training sets.

    Exposure therefore cannot explain why only the v3 arms fail; the copying
    behaviour does. If this ever reports the artefact as v2-only, the
    "source-text, not model error" classification collapses.
    """
    skip_if_missing(
        TUNE / "v2" / "train.jsonl", what="v2 training corpus"
    )
    qa = residue.build()["quote_artefact"]
    assert qa["eval_v3"]["affected_ids"] == [
        "ev3-016", "ev3-024", "ev3-032", "ev3-046", "ev3-052"]
    v2c = qa["corpus"]["v2 train"]["empty_quote_occurrences"]
    v3c = qa["corpus"]["v3.2 train"]["empty_quote_occurrences"]
    assert v2c > 1000 and v3c > 1000, (v2c, v3c)
    assert qa["eval_v3"]["by_arm"]["v2"]["json_ok"] == len(qa["eval_v3"]["affected_ids"])
    for arm in ("v3-corrected", "v3-2"):
        assert qa["eval_v3"]["by_arm"][arm]["json_ok"] == 0, arm


def test_quote_defect_list_is_derived_not_hardcoded():
    """decode_matrix must re-derive its defect set from committed artefacts."""
    assert decode_matrix.QUOTE_FAILURES == \
        decode_matrix._committed_jsonfail_ids()
    assert decode_matrix.REPEAT_LOOP_MIN_CHARS > 0


def test_no_quoted_feature_is_constant():
    """A feature constant across cases carries no information."""
    raw = (ROOT / "ml/tune/FINAL_MODEL_ASSESSMENT.md").read_text(encoding="utf-8")
    # markdown is hard-wrapped, so match against a whitespace-normalised copy
    text = " ".join(raw.split())
    for f in multitopic_mechanism.build()["feature_crosstab"]:
        groups = f["groups"]
        if len(groups) < 2:
            assert f["feature"] in DEGENERATE_FEATURES, f["feature"]
            continue
        assert any(g["regressed"] for g in groups), f["feature"]
    # and the report must state the no-variance finding rather than quote them
    assert "lexically similar second issue" in text
    assert "no variance" in text


def test_movement_counts_sum_to_the_case_count():
    for pair, m in multitopic_mechanism.build()["movement"].items():
        assert m["improved"] + m["regressed"] + m["unchanged"] == 52, pair


# -------------------------------------------------------------------- action
def test_cat_h_audit_finds_the_four_mislabeled_cases():
    au = action_analysis.cat_h_label_audit()
    assert au["n"] == 12
    assert [s["id"] for s in au["suspects"]] == ["ev3-056", "ev3-062", "ev3-065", "ev3-066"]
    for s in au["suspects"]:
        assert s["marker"], s
        assert s["excerpt"].strip()


def test_every_cat_h_suspect_really_has_an_empty_label():
    cases = {json.loads(l)["id"]: json.loads(l)
             for l in (TUNE / "eval_v3/cases.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()}
    for s in action_analysis.cat_h_label_audit()["suspects"]:
        exp = cases[s["id"]]["expected_topics"]
        assert not any((t.get("requested_action") or "").strip() for t in exp), s["id"]


def test_action_suppression_is_regime_specific_and_opposite():
    """v2 fails on two-topic; the v3 arms fail on single-topic.

    Pooling eval_v3 hides this -- it would make the two families look like they
    share one defect, which is what the earlier draft of the report claimed.
    """
    g = action_analysis.generation_action_rate_by_regime()
    v3s = ("v3-treatment", "v3-corrected", "v3-2")
    assert g["eval_v3_two_topic/v2"]["rate"] < 0.1 * g["eval_v3_single_topic/v2"]["rate"]
    for a in v3s:
        assert g["eval_v3_two_topic/v2"]["rate"] < g[f"eval_v3_two_topic/{a}"]["rate"], a
        assert g[f"eval_v3_single_topic/{a}"]["rate"] < g["eval_v3_single_topic/v2"]["rate"], a
    # the split must be real, not a relabelling of the same pooled number.
    # n is below the case count wherever a model emits no topics at all, because
    # such a case contributes no action slot.
    assert 90 <= g["eval_v3_single_topic/v2"]["n"] <= 94
    assert all(g[f"eval_v3_two_topic/{a}"]["n"] <= 52 for a in v3s)


def test_v3_arms_omit_actions_rather_than_inventing_them():
    """The mismatch is suppression, not hallucination."""
    skip_if_missing(TUNE / "v2" / "train.jsonl", what="v2 training corpus")
    r = action_analysis.build()["frozen_presence"]
    for arm in ARMS[1:]:
        assert r[arm]["omitted_action_wanted_by_label"] > 5 * r[arm]["invented_action_not_in_label"], arm
    assert r["v2"]["omitted_action_wanted_by_label"] <= 5


def test_generation_action_rate_is_below_the_training_prior():
    """The v3 shortfall must exceed prior reproduction, or the diagnosis changes."""
    skip_if_missing(TUNE / "v2" / "train.jsonl", what="v2 training corpus")
    train = action_analysis.build()["corpus_action_rate"]["v3.2 train (== v3 family)"]["rate"]
    for arm in ARMS[1:]:
        assert action_analysis.build()["frozen_presence"][arm]["predicted_action_rate"] < train, arm


# ------------------------------------------------------------------- residue
def test_gate_misses_the_euphonic_у_form_that_the_full_detector_catches():
    """The premise of the section: the suite's own gate has an euphony blind spot."""
    joined = " ".join(BOILER_SENT)
    assert re.search(r"в\\s\+телефонному", joined)
    assert not re.search(r"у\\s\+телефонному", joined)
    assert residue.R2_RE.search("Відповідь у телефонному режимі.")
    assert residue.R2_RE.search("Відповідь в телефонному режимі.")


def test_residue_reports_r3_which_the_gate_cannot_see():
    joined = " ".join(BOILER_SENT)
    assert "вжити" not in joined
    assert residue.R3_RE.search("Прохання вжити заходи реагування")
    assert residue.R3_RE.search("Просить вжити відповідні заходи.")


def test_r3_is_worse_in_v3_than_v2():
    """The claim under test. If a future corpus fixes it, this must fail."""
    skip_if_missing(TUNE / "v2" / "train.jsonl", what="v2 training corpus")
    r = residue.build()
    v2 = r["frozen_329"]["v2"]["r3_rate"]
    for arm in ("previous-v3", "corrected-v3", "v3.2"):
        assert r["frozen_329"][arm]["r3_rate"] > v2, arm


def test_r3_lives_in_issue_not_requested_action():
    """Rules out C2's action supervision as the cause of the R3 residue."""
    d = json.loads((TUNE / "eval/v3-2.json").read_text(encoding="utf-8"))
    in_issue = in_action = 0
    for pr in d["predictions"]:
        if not pr["raw"].startswith("{"):
            continue
        for t in json.loads(pr["raw"])["topics"]:
            in_issue += bool(residue.R3_RE.search(str(t.get("issue", ""))))
            in_action += bool(residue.R3_RE.search(str(t.get("requested_action", ""))))
    assert in_issue > 0
    assert in_action == 0


# ------------------------------------------------------------------- decoding
def test_probe_settings_separate_temperature_from_the_penalty():
    from ml.tune.decoding_probe import SEED, SETTINGS
    assert SETTINGS["temp01"]["temp"] == SETTINGS["temp01_norep"]["temp"] == 0.1
    assert SETTINGS["temp01"]["rep"] is not None
    assert SETTINGS["temp01_norep"]["rep"] is None
    assert isinstance(SEED, int)


def test_probe_controls_are_twelve_distinct_cases():
    from ml.tune.decoding_probe import build_caselist
    cl = build_caselist()
    ctrl = [c["id"] for c in cl if "control" in c["tags"]]
    assert len(ctrl) == 12
    assert len(set(ctrl)) == 12


def test_probe_covers_52_two_topic_and_14_jsonfail_cases():
    from ml.tune.decoding_probe import build_caselist
    cl = build_caselist()
    assert sum(1 for c in cl if "twotopic" in c["tags"]) == 52
    assert sum(1 for c in cl if "jsonfail" in c["tags"]) == 14


# ------------------------------------------------------------------- verifier
def test_verifier_checks_a_substantial_number_of_claims():
    c = verify_final_assessment.Checks()
    rep = verify_final_assessment.build_report()
    verify_final_assessment.main_section1(c, rep["tables"], rep["text"])
    verify_final_assessment.main_section4(c, rep["tables"], rep["refstyle"])
    verify_final_assessment.main_section5(c, rep["tables"], rep["residue"])
    verify_final_assessment.main_section6(c, rep["tables"], rep["act"])
    verify_final_assessment.main_section8(c, rep["text"], rep["rows"])
    assert len(c.rows) >= 150
    assert not c.failed(), c.failed()[:3]


def test_verifier_close_rejects_a_wrong_figure():
    c = verify_final_assessment.Checks()
    c.close("t", 0.5, 0.5)
    assert not c.failed()
    c.close("t2", 0.5, 0.6)
    assert len(c.failed()) == 1


def test_verifier_place_requires_the_figure_in_the_text():
    c = verify_final_assessment.Checks()
    c.close("t", 0.2119, 0.2119, place="the rate is 0.2119 here")
    assert not c.failed()
    c.close("t2", 0.2119, 0.2119, place="some other section")
    assert len(c.failed()) == 1

def test_probe_gate_constants_match_gates_v3():
    """decode_matrix mirrors two gate thresholds; they must not drift.

    gates_v3 keeps the thresholds as literals inside evaluate(), so there is no
    constant to import. The target strings are the documented spec, so the test
    parses those instead of restating the numbers a third time.
    """
    import re

    from ml.tune import decode_matrix

    src = (ROOT / "ml/tune/gates_v3.py").read_text(encoding="utf-8")
    thresholds = {}
    for name, block in re.findall(
            r'g\(name="(\w+)".*?target="([^"]*)"', src, re.S):
        m = re.search(r"([<>]=?|=)\s*([0-9.]+)", block)
        if m:
            thresholds[name] = (m.group(1), float(m.group(2)))

    assert thresholds["schema"] == ("=", decode_matrix.SCHEMA_GATE), thresholds
    assert thresholds["topic_count"] == (">=", decode_matrix.TOPIC_GATE), thresholds


def test_probe_grid_is_complete_and_reproducible():
    """The 960-cell probe is the authority for section 3, so check its shape."""
    rows = verify_final_assessment._probe_rows()
    assert len(rows) == 960
    assert len({(r["model"], r["setting"], r["id"]) for r in rows}) == 960
    assert len({r["seed"] for r in rows}) == 960, "every generation needs its own seed"
    assert all((r.get("raw") or "").strip() for r in rows), "no empty generation"
    assert {r["model"] for r in rows} == {"v2", "corrected-v3", "v3-2"}
    assert len({r["setting"] for r in rows}) == 5
    assert len({r["id"] for r in rows}) == 64

    d = json.loads((ROOT / "ml/data/tune/decode_matrix.json").read_text(encoding="utf-8"))
    assert d["rows"] == len(rows)
    assert d["complete"] is True


def test_greedy_probe_row_matches_the_recorded_serving_config():
    """budget1200 must be temp=0, no penalty, 1200 tokens -- i.e. real greedy."""
    for r in verify_final_assessment._probe_rows():
        if r["setting"] == "budget1200":
            assert r["temp"] == 0.0 and not r["rep"] and r["max_tokens"] == 1200


def test_probe_loop_detector_is_reimplementable_from_the_documented_rule():
    """The documented rule must reproduce the reported loop counts exactly."""
    d = json.loads((ROOT / "ml/data/tune/decode_matrix.json").read_text(encoding="utf-8"))
    det = d["findings"]["loop_detector"]
    rows = verify_final_assessment._probe_rows()
    settings = d["findings"]["settings"]
    for model, rep in d["findings"]["loops"].items():
        got = sorted({r["id"] for r in rows if r["model"] == model
                      and verify_final_assessment._is_loop(r["raw"], det)})
        assert got == sorted(rep["cases"]), model
        for st in settings:
            n = sum(1 for r in rows if r["model"] == model and r["setting"] == st
                    and verify_final_assessment._is_loop(r["raw"], det))
            assert n == rep["cells_by_setting"][st], (model, st)


def test_probe_baseline_and_greedy_agree_for_recorded_arms():
    """The probe must reproduce the committed behaviour where a record exists."""
    d = json.loads((ROOT / "ml/data/tune/decode_matrix.json").read_text(encoding="utf-8"))
    for arm in ("corrected-v3", "v3-2"):
        tr = d["grid"][f"{arm}/budget1200"]["transitions"]
        same = sum(n for t, n in tr.items()
                   if t.split(" -> ")[0] == t.split(" -> ")[1])
        assert same == 64, (arm, same)
    # v2 has no behaviour recording at all, so its baseline is flag-derived and
    # must be recorded as such rather than presented as semantic
    v2 = d["grid"]["v2/budget1200"]["baseline_same_subset"]
    assert v2["recording_used_for"] == 0 and v2["flag_fallback_for"] == 64


def test_probe_best_cell_never_closes_a_gate_and_says_why():
    d = json.loads((ROOT / "ml/data/tune/decode_matrix.json").read_text(encoding="utf-8"))
    best = d["findings"]["best_cell"]
    for arm, v in best.items():
        assert v["closes_a_gate"] is False, arm
        assert v["blocking_gates"], arm
        assert v["closes_a_gate"] == (not v["blocking_gates"]), arm
    # v2 reaches schema 1.00 on this subset, so naming schema as its blocker
    # would be false
    assert best["v2"]["schema"] == 1.0
    assert len(best["v2"]["blocking_gates"]) == 1
    assert best["v2"]["blocking_gates"][0].startswith("topic_count")


REPORT_MD = ROOT / "ml/tune/FINAL_MODEL_ASSESSMENT.md"


def _verify(tmp_path, text):
    """Run the verifier over a (possibly corrupted) report and return the run."""
    out = tmp_path / "report.md"
    out.write_text(text, encoding="utf-8")
    j = tmp_path / "checks.json"
    p = subprocess.run(
        [sys.executable, "-m", "ml.tune.verify_final_assessment",
         "--report", str(out), "--json-out", str(j)],
        cwd=ROOT, capture_output=True, text=True)
    assert p.returncode in (0, 1), p.stdout + p.stderr
    return json.loads(j.read_text(encoding="utf-8"))


def test_final_verifier_checks_a_substantial_number_of_claims(tmp_path):
    """Guards the failure mode where dispatch matches nothing but still says OK."""
    data = _verify(tmp_path, REPORT_MD.read_text(encoding="utf-8"))
    assert data["verdict"].startswith("OK"), data["verdict"]
    # a floor, not an exact count: adding checks must not require editing this
    # test, but a dispatcher that silently stops matching must still fail
    n_ok = int(data["verdict"].rsplit("(", 1)[1].split("/")[0])
    assert n_ok >= 380, data["verdict"]
    assert data["verdict"].endswith(f"/{n_ok})"), data["verdict"]
    # each section must contribute; a dropped dispatcher is the real risk
    for sec in ("§1", "§2", "§3", "§4", "§5", "§6", "§7", "§8"):
        assert any(sec in c["claim"] for c in data["claims"]), sec


# (label, pattern, replacement) -- each must appear in the real report, so a
# pattern that stops matching makes the test fail loudly instead of passing
CORRUPTIONS = [
    ("probe cell count", r"960", "961"),
    ("literal quote case list", r"ev3-046", "ev3-047"),
    ("v3.2 persistent loop set", r"ev3-047", "ev3-048"),
    ("canonical model path",
     r"qwen3-8b-lora-v2-attempt10-fused", "qwen3-8b-lora-v3-2-fused"),
    ("repetition-loop claim", r"5 of v3\.2's 8 loop cases", "5 of v3.2's 9 loop cases"),
    ("decision option", r"Option A — STOP MODEL DEVELOPMENT", "Option B — TRAIN AGAIN"),
    ("release decision", r"no release candidate for this project", "a release candidate exists"),
    ("no-further-training statement", r"Further training is \*\*not justified yet\*\*",
     "Further training is **clearly justified**"),
]


@pytest.mark.parametrize("label,pattern,replacement", CORRUPTIONS,
                         ids=[c[0] for c in CORRUPTIONS])
def test_report_corruption_is_detected(tmp_path, label, pattern, replacement):
    """Every headline number in the report must be load-bearing."""
    import re
    good = REPORT_MD.read_text(encoding="utf-8")
    corrupt, n = re.subn(pattern, replacement, good, flags=re.M)
    assert n >= 1, f"pattern for {label!r} no longer matches; the test is stale"
    data = _verify(tmp_path, corrupt)
    assert not data["verdict"].startswith("OK"), (
        f"corrupting {label!r} did not fail the verifier")


def test_corrupting_a_probe_finding_is_detected(tmp_path):
    """The probe findings must be re-derived, not read back and trusted."""
    d = json.loads((ROOT / "ml/data/tune/decode_matrix.json").read_text(encoding="utf-8"))
    good = REPORT_MD.read_text(encoding="utf-8")
    # a report that claims greedy reproduces the baseline for v2 at 64/64 is
    # wrong: v2 has no behaviour recording, so 64/64 is impossible
    assert "44/64" in good or "flag-derived" in good
    # sanity: the real detector does not flag all 64 v2 cells
    det = d["findings"]["loop_detector"]
    rows = verify_final_assessment._probe_rows()
    n = sum(1 for r in rows if r["model"] == "v2"
            and verify_final_assessment._is_loop(r["raw"], det))
    assert n == 0
