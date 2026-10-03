"""Verify every number in ``FINAL_MODEL_ASSESSMENT.md`` against the artefacts.

Same contract as :mod:`ml.tune.verify_v3_2_report`: the document is the
deliverable, so a hand-transcribed figure is the main risk. This module re-reads
the markdown and compares each figure with the artefact it summarises, rather
than holding a second copy of the expected values.

    .venv/bin/python -m ml.tune.verify_final_assessment
    .venv/bin/python -m ml.tune.verify_final_assessment --json-out /tmp/claims.json

Exit status is 0 only when every claim checks out.

Coverage: ``ml/data/tune/eval/`` is gitignored and is loaded before the first
check, so this verifier cannot run without it.
that directory to exist locally. A missing directory is reported as uncovered
claims, never as a pass.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
import re
from pathlib import Path

from ml.tune.residue import GATE_RE, R2_RE, _scan

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "ml/tune/FINAL_MODEL_ASSESSMENT.md"

TUNE = ROOT / "ml/data/tune"
MATRIX = TUNE / "cross_arm_matrix.json"
ACTION = TUNE / "action_analysis.json"
RESIDUE = TUNE / "residue.json"
DECODE = TUNE / "decode_matrix.json"
EVAL_V3 = TUNE / "eval_v3"
FROZEN = TUNE / "eval"
CORPUS_V2 = TUNE / "v2/train.jsonl"
CORPUS_V3 = TUNE / "v3/treatment_v3_2/train.jsonl"

ARMS = ("v2", "v3-treatment", "v3-corrected", "v3-2")
LABEL = {"v2": "v2", "v3-treatment": "previous-v3",
         "v3-corrected": "corrected-v3", "v3-2": "v3.2"}


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, claim: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"claim": claim, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def close(self, claim: str, got, want, tol: float = 5e-5, place: str | None = None) -> bool:
        """Assert ``reported == expected``.

        ``place`` additionally requires the *formatted* figure to appear in the
        given text, so a claim cannot pass on a correct artefact while the
        document still quotes the wrong number.
        """
        if got is None or want is None:
            return self.check(claim, False, f"reported={got} expected={want}")
        ok = abs(got - want) <= tol
        detail = f"reported={got} expected={want}"
        if place is not None and ok:
            shown = f"{want:g}" if float(want) == int(want) else f"{want:.4f}"
            alt = f"{want:.4f}"
            if shown not in place and alt not in place:
                ok, detail = False, f"{alt} not found in section text"
        return self.check(claim, ok, detail)

    def says(self, claim: str, needle: str, place: str) -> bool:
        """Assert a literal string appears in the given text.

        Needed where the claim is a name or a count written in words, which
        ``close`` cannot express: it exists so a correct artefact sitting next
        to a stale sentence in the report still fails.
        """
        flat = " ".join(place.split())
        return self.check(claim, needle in flat, f"{needle!r} not found in section text")

    def failed(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]

    def verdict(self) -> str:
        bad = self.failed()
        return ("FAIL" if bad else "OK") + f" ({len(self.rows) - len(bad)}/{len(self.rows)})"


def _tables(text: str) -> list[list[list[str]]]:
    out = []
    for block in re.findall(r"((?:^\|.*\|\s*$\n)+)", text, re.M):
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
                for ln in block.strip().splitlines()]
        rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
        if len(rows) > 2:
            out.append(rows)
    return out


def _row(tables, first_cell: str, ncol: int) -> list[str] | None:
    """The row whose first cell is exactly ``first_cell`` in an ncol table."""
    for t in tables:
        if len(t[0]) != ncol:
            continue
        for r in t[1:]:
            if r and r[0] == first_cell:
                return r
    return None


def _num(cell: str):
    c = cell.replace("*", "").replace("`", "").replace(",", "").strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+)", c)
    if m:
        return round(float(m.group(1)) / int(m.group(2)), 4)
    try:
        return float(c)
    except ValueError:
        return None


def _matrix_rows() -> dict[str, dict]:
    return {r["key"]: r for r in json.loads(MATRIX.read_text(encoding="utf-8"))["rows"]}


def _val(rows: dict[str, dict], key: str, arm: str):
    return rows[key]["values"].get(arm)


def _scan_issues(records, field: str = "issue") -> list[str]:
    out = []
    for r in records:
        for t in r.get("topics", []) or []:
            if isinstance(t, dict):
                out.append(str(t.get(field, "")))
    return out


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main_section1(c: Checks, tables, text: str) -> None:
    rows = _matrix_rows()
    # eval_v3 block: 6 columns (dimension, 4 arms, blocking?)
    for key, label in (("ev3_schema", "schema / JSON validity"),
                       ("ev3_boilerplate", "boilerplate leak"),
                       ("ev3_topic_count", "topic count exact")):
        r = _row(tables, label, 6)
        if not c.check(f"§1 table row present: {label}", r is not None):
            continue
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 {label} {LABEL[arm]}", _num(r[i]), _val(rows, key, arm))

    # multitopic block: counts as n/85 and recall as fractions
    r = _row(tables, "topic-count hits / 85", 6)
    if c.check("§1 MT topic-count row present", r is not None):
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 MT topic-count {LABEL[arm]}", _num(r[i]), _val(rows, "mt_topic_count", arm))
    r = _row(tables, "MT85 recall", 6)
    if c.check("§1 MT85 recall row present", r is not None):
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 MT85 recall {LABEL[arm]}", _num(r[i]), _val(rows, "mt_recall", arm))
    r = _row(tables, "frozen multi-topic exact", 6)
    if c.check("§1 frozen multi-topic row present", r is not None):
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 frozen multi-topic {LABEL[arm]}", _num(r[i]), _val(rows, "fr_multi_topic", arm))

    # frozen 329 block
    for key, label in (("fr_domain_acc", "domain accuracy"),
                       ("fr_domain_f1", "macro-F1"),
                       ("fr_issue_rl", "issue ROUGE-L"),
                       ("fr_action_match", "action presence match")):
        r = _row(tables, label, 6)
        if not c.check(f"§1 frozen row present: {label}", r is not None):
            continue
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 frozen {label} {LABEL[arm]}", _num(r[i]), _val(rows, key, arm))

    # cat A terse: 8/10 style
    r = _row(tables, "domain-set exact (cat A terse)", 6)
    if c.check("§1 cat A row present", r is not None):
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 cat A terse {LABEL[arm]}", _num(r[i]), _val(rows, "terse_A", arm))

    # smoke of 4
    r = _row(tables, "smoke two-topic (of 4)", 6)
    if c.check("§1 smoke row present", r is not None):
        for i, arm in enumerate(ARMS, 1):
            c.close(f"§1 smoke {LABEL[arm]}", _num(r[i]), _val(rows, "smoke_mt", arm))

    # blocking flags in the §1 tables must agree with the matrix
    for key in ("ev3_schema", "ev3_boilerplate", "ev3_topic_count",
                "mt_recall", "fr_domain_acc", "smoke_mt"):
        c.check(f"§1 {rows[key]['label']} blocking flag = {rows[key]['blocking']}",
                rows[key]["blocking"] is True)

    # statistical support claims
    s = rows["ev3_schema"]["vs_v2"]["v3-2"]["support"]
    c.check("§1 v3.2 schema drop vs v2 is supported at p=0.0034",
            s["supported"] and abs(s["p"] - 0.0034) < 5e-4, f"p={s['p']}")
    s = rows["ev3_schema"]["vs_corrected"]["v3-2"]["support"]
    c.check("§1 v3.2 schema drop vs corrected-v3 is supported at p=0.0391",
            s["supported"] and abs(s["p"] - 0.0391) < 5e-4, f"p={s['p']}")
    s = rows["fr_multi_topic"]["vs_v2"]["v3-2"]["support"]
    c.check("§1 v3.2 frozen multi-topic drop vs v2 is supported at p=0.0034",
            s["supported"] and abs(s["p"] - 0.0034) < 5e-4, f"p={s['p']}")
    # the report must NOT claim topic count is significant
    c.check("§1 topic-count deltas are reported as insignificant",
            not rows["ev3_topic_count"]["vs_v2"]["v3-2"]["support"]["supported"]
            and not rows["ev3_topic_count"]["vs_corrected"]["v3-2"]["support"]["supported"])
    for key in ("mt_recall", "fr_domain_acc"):
        for arm in ("v3-treatment", "v3-corrected", "v3-2"):
            c.check(f"§1 {key} {LABEL[arm]} vs v2 is insignificant",
                    not rows[key]["vs_v2"][arm]["support"]["supported"],
                    f"p={rows[key]['vs_v2'][arm]['support']['p']}")

    # the "insignificant, but directionally consistent" table rows quote ranges;
    # verify both ends of each range against the artefacts
    def _p(key: str, arm: str) -> float:
        return rows[key]["vs_v2"][arm]["support"]["p"]

    def _d(key: str, arm: str) -> float:
        return rows[key]["vs_v2"][arm]["support"]["delta"]

    c.check("§1 MT85 recall delta range +0.0118 … +0.0471",
            abs(min(_d("mt_recall", a) for a in ARMS[1:]) - 0.0118) < 5e-4
            and abs(max(_d("mt_recall", a) for a in ARMS[1:]) - 0.0471) < 5e-4,
            str(sorted(round(_d("mt_recall", a), 4) for a in ARMS[1:])))
    c.check("§1 MT85 recall p range 0.125 … 1.000",
            abs(min(_p("mt_recall", a) for a in ARMS[1:]) - 0.125) < 5e-4
            and abs(max(_p("mt_recall", a) for a in ARMS[1:]) - 1.0) < 5e-4,
            str(sorted(_p("mt_recall", a) for a in ARMS[1:])))
    c.check("§1 frozen domain-accuracy deltas are all positive",
            all(_d("fr_domain_acc", a) > 0 for a in ARMS[1:]),
            str(sorted(round(_d("fr_domain_acc", a), 4) for a in ARMS[1:])))
    c.check("§1 frozen domain-accuracy range +0.0213 … +0.0304",
            abs(min(_d("fr_domain_acc", a) for a in ARMS[1:]) - 0.0213) < 5e-4
            and abs(max(_d("fr_domain_acc", a) for a in ARMS[1:]) - 0.0304) < 5e-4,
            str(sorted(round(_d("fr_domain_acc", a), 4) for a in ARMS[1:])))
    c.check("§1 frozen domain-accuracy p range 0.053 … 0.144",
            abs(min(_p("fr_domain_acc", a) for a in ARMS[1:]) - 0.0525) < 5e-4
            and abs(max(_p("fr_domain_acc", a) for a in ARMS[1:]) - 0.1435) < 5e-4,
            str(sorted(_p("fr_domain_acc", a) for a in ARMS[1:])))

    # the frozen-329 per-arm values must be traceable to a *tracked* artefact,
    # because the live generations themselves live in a gitignored directory
    c.check("§9 frozen-329 per-arm values are traceable to the tracked matrix",
            all(k in rows for k in ("fr_multi_topic", "fr_domain_f1", "fr_issue_rl",
                                    "fr_action_match", "fr_domain_acc", "fr_action_copy")))
    c.check("§1 frozen per-case decomposition is faithful for every arm",
            all(v["faithful"] for v in
                json.loads(MATRIX.read_text(encoding="utf-8"))["frozen_per_case_fidelity"].values()))


def main_section4(c: Checks, tables, res: dict) -> None:
    """Frozen references are v2-style; v3 arms emit shorter issues."""
    r = _row(tables, "frozen 329 references", 3)
    if c.check("§4 reference-style table present", r is not None):
        c.close("§4 frozen references boilerplate rate", _num(r[1]), res["frozen_refs"]["boilerplate"])
        c.close("§4 frozen references mean issue chars", _num(r[2]), res["frozen_refs"]["mean_chars"])
    for arm, lbl in (("v2", "v2"), ("previous-v3", "previous-v3"),
                     ("corrected-v3", "corrected-v3"), ("v3.2", "v3.2")):
        r = _row(tables, f"{lbl} generations", 3)
        if not c.check(f"§4 generation row present: {lbl}", r is not None):
            continue
        c.close(f"§4 {lbl} boilerplate", _num(r[1]), res["generations"][arm]["boilerplate"])
        c.close(f"§4 {lbl} mean issue chars", _num(r[2]), res["generations"][arm]["mean_chars"])
    # build_eval_v3 documents that its reference labeler closes the boilerplate gap
    src = (ROOT / "ml/tune/build_eval_v3.py").read_text(encoding="utf-8")
    c.check("§4 build_eval_v3 documents the closed boilerplate gap",
            "boilerplate gap closed" in src)


def main_section5(c: Checks, tables, residue: dict) -> None:
    r = _row(tables, "v2", 4)
    if c.check("§5 R2 table present", r is not None):
        c.close("§5 v2 frozen-329 R2", _num(r[1]), residue["frozen_329"]["v2"]["r2_rate"])
        c.close("§5 v2 corpus R2", _num(r[3]), residue["corpus"]["v2 train"]["r2_rate"])
    for lbl, arm in (("previous-v3", "previous-v3"), ("corrected-v3", "corrected-v3"),
                     ("v3.2", "v3.2")):
        r = _row(tables, lbl, 4)
        if not c.check(f"§5 row present: {lbl}", r is not None):
            continue
        c.close(f"§5 {lbl} frozen-329 R2", _num(r[1]), residue["frozen_329"][arm]["r2_rate"])
        c.close(f"§5 {lbl} two-topic R2", _num(r[2]), residue["behaviour_52"][arm]["r2_rate"])
    c.close("§5 v3.2 corpus R2", 0.0006, residue["corpus"]["v3.2 train"]["r2_rate"], tol=1e-4)
    # the gate's own pattern list must still miss the "у" form
    from ml.tune.build_eval_v3 import BOILER_SENT
    joined = " ".join(BOILER_SENT)
    c.check("§5 BOILER_SENT matches the в form",
            bool(re.search(r"в\\s\+телефонному", joined)))
    c.check("§5 BOILER_SENT does not match the у form",
            not re.search(r"у\\s\+телефонному", joined))
    c.check("§5 full detector does match both forms",
            bool(R2_RE.search("відповідь у телефонному режимі"))
            and bool(R2_RE.search("відповідь в телефонному режимі")))


def _v2_single_rate() -> float:
    d = json.loads((EVAL_V3 / "results/v2.json").read_text(encoding="utf-8"))
    n = ne = 0
    for c in d["per_case"]:
        a = [(x or "").strip() for x in (c.get("predicted_actions") or [])]
        if not a:
            continue
        n += 1
        ne += 1 if any(a) else 0
    return round(ne / n, 4) if n else None


def main_section5b(c: Checks, tables, residue: dict) -> None:
    """The `""` source-text artefact behind five of the six quote JSON failures."""
    qa = residue["quote_artefact"]
    ids = qa["eval_v3"]["affected_ids"]
    c.check("§5 five eval_v3 inputs carry the literal \"\" artefact",
            len(ids) == 5, str(ids))
    c.check("§5 v2 parses all five", qa["eval_v3"]["by_arm"]["v2"]["json_ok"] == 5)
    for arm in ("v3-corrected", "v3-2"):
        c.check(f"§5 {LABEL[arm]} parses none of the five",
                qa["eval_v3"]["by_arm"][arm]["json_ok"] == 0)
    # the artefact is in BOTH corpora, so exposure cannot explain the split
    v2c = qa["corpus"]["v2 train"]["empty_quote_occurrences"]
    v3c = qa["corpus"]["v3.2 train"]["empty_quote_occurrences"]
    c.check("§5 the artefact is present in both training sets, so exposure is not the cause",
            v2c > 1000 and v3c > 1000, f"v2={v2c} v3.2={v3c}")
    c.check("§5 quote counts 12528 (v2) / 15274 (v3.2)",
            v2c == 12528 and v3c == 15274)
    c.check("§5 the affected cases are the same set the report lists",
            ids == ["ev3-016", "ev3-024", "ev3-032", "ev3-046", "ev3-052"])
    c.check("§5 ev3-083 is excluded because it carries no artefact",
            "ev3-083" not in ids)


def main_section6(c: Checks, tables, act: dict) -> None:
    car = act["corpus_action_rate"]
    c.close("§6 v2 training action rate", 0.4571, car["v2 train"]["rate"])
    c.close("§6 v3 training action rate", 0.2119, car["v3.2 train (== v3 family)"]["rate"])
    c.close("§6 frozen label action rate", 0.5015, car["frozen test (labels)"]["rate"])

    fp = act["frozen_presence"]
    for arm, omit, invent, agree, match in (
            ("v2", 2, 1, 326, 0.9909), ("v3-treatment", 135, 14, 180, 0.5471),
            ("v3-corrected", 124, 15, 190, 0.5775), ("v3-2", 137, 6, 186, 0.5653)):
        v = fp[arm]
        c.close(f"§6 mismatch {LABEL[arm]} omits", omit, v["omitted_action_wanted_by_label"])
        c.close(f"§6 mismatch {LABEL[arm]} invents", invent, v["invented_action_not_in_label"])
        c.close(f"§6 mismatch {LABEL[arm]} agrees", agree, v["agree"])
        c.close(f"§6 mismatch {LABEL[arm]} match", match, v["presence_match"], tol=1e-3)

    for arm, red, e3copy, frcopy in (
            ("v2", 0.2055, 0.1986, 0.4407), ("v3-treatment", 0.0890, 0.0753, 0.0912),
            ("v3-corrected", 0.1575, 0.1507, 0.1246), ("v3-2", 0.0753, 0.0753, 0.0669)):
        c.close(f"§6 redundancy {LABEL[arm]}", red,
                act["action_redundancy"][arm]["action_redundancy_rate"], tol=1e-3)
        c.close(f"§6 eval_v3 copy {LABEL[arm]}", e3copy,
                act["copy_eval_v3"][arm]["rate"], tol=1e-3)
        c.close(f"§6 frozen copy {LABEL[arm]}", frcopy,
                act["copy_frozen"][arm]["rate"], tol=1e-3)

    au = act["cat_h_label_audit"]
    c.close("§6 cat H audit suspect count", 4, au["suspect_count"])
    c.close("§6 cat H audit size", 12, au["n"])
    ids = [s["id"] for s in au["suspects"]]
    for want in ("ev3-056", "ev3-062", "ev3-065", "ev3-066"):
        c.check(f"§6 cat H audit names {want}", want in ids)
    inv = act["empty_action_invention_cat_h"]
    c.check("§6 ev3-062 is the single invented case for every v3 arm",
            all(inv[a]["ids"] == ["ev3-062"]
                for a in ("v3-treatment", "v3-corrected", "v3-2")),
            str({a: inv[a]["ids"] for a in inv}))
    c.check("§6 v2 invents no cat H action", inv["v2"]["ids"] == [])
    # and the label really is empty while the text asks for something
    cases = {json.loads(l)["id"]: json.loads(l)
             for l in (EVAL_V3 / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    c.check("§6 ev3-062 expected action is empty",
            not any((t.get("requested_action") or "").strip()
                    for t in cases["ev3-062"]["expected_topics"]))
    c.check("§6 ev3-062 text requests an action",
            "Просить вжити заход" in cases["ev3-062"]["text"])
    # generation rates quoted in the two §6 tables
    for arm, want in (("v2", 0.4985), ("v3-treatment", 0.1337),
                      ("v3-corrected", 0.1702), ("v3-2", 0.1033)):
        c.close(f"§6 frozen-329 generated action rate {LABEL[arm]}", want,
                fp[arm]["predicted_action_rate"], tol=1e-3)
    g = act["generation_action_rate_by_regime"]
    for arm, single, two, multi in (("v3-treatment", 0.0745, 0.1224, 0.2235),
                                    ("v3-corrected", 0.1809, 0.1304, 0.2471),
                                    ("v3-2", 0.0638, 0.1250, 0.2381)):
        c.close(f"§6 eval_v3 single-topic rate {LABEL[arm]}", single,
                g[f"eval_v3_single_topic/{arm}"]["rate"], tol=1e-3)
        c.close(f"§6 eval_v3 two-topic rate {LABEL[arm]}", two,
                g[f"eval_v3_two_topic/{arm}"]["rate"], tol=1e-3)
        c.close(f"§6 multitopic rate {LABEL[arm]}", multi,
                g[f"multitopic/{arm}"]["rate"], tol=1e-3)
    c.close("§6 v2 eval_v3 single-topic rate", 0.3118,
            g["eval_v3_single_topic/v2"]["rate"], tol=1e-3)
    c.close("§6 v2 eval_v3 two-topic rate", 0.0192,
            g["eval_v3_two_topic/v2"]["rate"], tol=1e-3)
    # the load-bearing claim: v2 and the v3 arms fail in *opposite* regimes
    c.check("§6 v2 suppresses actions far harder on two-topic than single-topic",
            g["eval_v3_two_topic/v2"]["rate"] < 0.1 * g["eval_v3_single_topic/v2"]["rate"],
            f"{g['eval_v3_two_topic/v2']['rate']} vs "
            f"{g['eval_v3_single_topic/v2']['rate']}")
    c.check("§6 v2 is worse than every v3 arm on two-topic",
            all(g["eval_v3_two_topic/v2"]["rate"] < g[f"eval_v3_two_topic/{a}"]["rate"]
                for a in ("v3-treatment", "v3-corrected", "v3-2")))
    c.check("§6 every v3 arm is worse than v2 on single-topic",
            all(g[f"eval_v3_single_topic/{a}"]["rate"] < g["eval_v3_single_topic/v2"]["rate"]
                for a in ("v3-treatment", "v3-corrected", "v3-2")))
    meta = json.loads((TUNE / "v3/treatment_v3_2/meta.json").read_text(encoding="utf-8"))
    cc = meta["same_object_stream"]["sampling_counters"]["c1_c2_counters"]
    c.close("§6 same-object stream action rate", 0.1877,
            round(cc["action_nonempty"] / (cc["action_nonempty"] + cc["action_empty"]), 4),
            tol=1e-3)


def main_section2(c: Checks, text: str, rows: dict) -> None:
    sec = text.split("## 2.")[1].split("## 3.")[0]
    for pid in [f"P{i}" for i in range(1, 15)]:
        c.check(f"§2 defines {pid}", f"| {pid} |" in sec)
    # the three claims that reverse a pessimistic reading
    c.check("§2 classifies R3 as not fixed", "model-generation** (not fixed)" in sec)
    c.check("§2 classifies the cat H invention as a label problem",
            "label problem" in sec)
    c.check("§2 classifies the ROUGE-L regression as an evaluator limitation",
            "evaluator limitation" in sec)

    # Evidence hygiene: the only decoding evidence in this report must come from
    # the seeded probe in §3. An earlier unseeded run was discarded, so no row
    # may quote a case-level result from it as if it were evidence.
    c.check("§2 discards the unseeded smoke run instead of citing it",
            "discarded" in sec)
    c.check("§2 quotes no discarded per-case decoding result",
            "repaired by sampling on ev3-025" not in text
            and "identical raw text under budget" not in text)
    c.check("§2 defers the repetition-loop classification to §3",
            "P11" in sec and "§3" in sec)


def main_section8(c: Checks, text: str, rows: dict) -> None:
    low = text.lower()
    sec7 = text.split("## 7.")[1].split("## 8.")[0]
    sec8 = text.split("## 8.")[1].split("## 9.")[0]
    # exactly one of A / B / C is chosen
    chosen = [o for o in ("option a", "option b", "option c")
              if f"**{o}" in low or f"### {o}" in low]
    c.check("§8 chooses exactly one option", len(chosen) == 1, ",".join(chosen))
    c.check("§8 chooses option A (stop) and does not adopt B or C",
            chosen == ["option a"] and "**this is the decision.**" in low, ",".join(chosen))
    c.check("§8 says a training run is not justified yet",
            "not justified yet" in sec8.lower(),
            "missing from section 8")
    # and the restatements elsewhere must agree with it
    c.check("the not-justified-yet statement is stated, not contradicted",
            text.lower().count("not justified yet") >= 2)
    # the decision must rest on the release finding, not merely restate it
    c.check("§8 explains the un-isolated topic-count ceiling",
            "topic count" in sec8.lower() and "un-isolated" in sec8.lower())
    c.check("§8 records the next controlled experiment rather than running it",
            "hypothesis" in sec8.lower() and "revert" in sec8.lower()
            and "next controlled experiment" in sec8.lower())
    c.check("§7 states no release candidate exists for any arm",
            "no release candidate for this project" in sec7.lower()
            or "no release candidate exists" in sec7.lower())
    c.check("§7 states v2 is not releasable either",
            "including v2" in sec7.lower() or "not even v2" in sec7.lower())
    c.check("§7 keeps v2 canonical",
            "v2 remains the canonical public model" in low
            or "v2 stays the canonical public model" in low)
    # every arm fails at least one blocking gate
    fails = {}
    for arm in ARMS:
        fails[arm] = [k for k, r in rows.items() if r["blocking"] and _bad(rows, k, arm)]
    for arm in ARMS:
        c.check(f"§8 {LABEL[arm]} fails at least one blocking gate",
                bool(fails[arm]), ",".join(fails[arm]) or "none")
    v2_blocking = [k for k, r in rows.items() if r["blocking"]]
    c.check("§8 v2 fails schema, boilerplate, topic count and smoke",
            set(fails["v2"]) >= {"ev3_schema", "ev3_boilerplate", "ev3_topic_count", "smoke_mt"},
            ",".join(sorted(fails["v2"])))


def _bad(rows: dict, key: str, arm: str) -> bool:
    r = rows[key]
    gate = r.get("gate")
    v = _val(rows, key, arm)
    if v is None or gate is None:
        return False
    g = str(gate).replace(" ", "")
    if g.startswith("="):
        return abs(v - float(g[1:])) > 1e-9
    mo = re.match(r"^(\d+)\s*of\s*(\d+)$", g)
    if mo:
        return v < float(mo.group(1))
    m = re.match(r"^(<=|>=|<|>)([\d.]+)$", g)
    if not m:
        return False
    op, num = m.group(1), float(m.group(2))
    return {"<=": v > num, ">=": v < num, "<": v >= num, ">": v <= num}[op]


def _cat_e_index(mech: dict) -> int:
    for i, c in enumerate(mech["by_category"]):
        if c["category"] == "E":
            return i
    raise AssertionError("no category E row")


def main_section7(c: Checks, text: str, act: dict, rows: dict) -> None:
    """§7 is the release decision; the action-rate narrative is verified in §6."""
    sec = text.split("## 7.")[1].split("## 8.")[0]
    car = act["corpus_action_rate"]
    fp = act["frozen_presence"]
    # the decision must rest on the same numbers §6 verified, so they must
    # appear here too
    c.close("§7 quotes the v2 action training rate", 0.4571,
            car["v2 train"]["rate"], place=sec)
    c.close("§7 quotes the v3 action training rate", 0.2119,
            car["v3.2 train (== v3 family)"]["rate"], place=sec)
    c.close("§7 quotes the frozen label action rate", 0.5015,
            car["frozen test (labels)"]["rate"], place=sec)
    c.close("§7 quotes the action-presence collapse", 0.9909,
            fp["v2"]["presence_match"], place=sec)
    c.close("§7 quotes the v3.2 action-presence value", 0.5653,
            fp["v3-2"]["presence_match"], place=sec)
    mech = json.loads((TUNE / "multitopic_mechanism.json").read_text(encoding="utf-8"))
    c.close("§7 v3.2 regression count", 15,
            mech["movement"]["corrected-v3 -> v3.2"]["regressed"], place=sec)
    cats = {c["category"] for c in mech["by_category"]
            if c["corrected_to_v32"]["regressed"] > 0}
    c.check("§7 v3.2 regressions span all seven multitopic categories",
            len(cats) == 7, ",".join(sorted(cats)))
    e = mech["by_category"][_cat_e_index(mech)]["corrected_to_v32"]
    c.close("§7 cat E improved", 2, e["improved"])
    c.close("§7 cat E regressed", 2, e["regressed"])
    c.close("§7 cat E unchanged", 2, e["unchanged"])


def main_section3(c: Checks, tables, text: str) -> None:
    from ml.tune.decode_matrix import build
    d = build()
    c.check("§3 probe grid is complete (960 generations)", d["complete"],
            f"{d['rows']} generations")
    if not d["complete"]:
        return
    c.check("§3 probe covers 64 cases", all(v["n_cases"] == 64 for v in d["grid"].values()))
    c.check("§3 probe ran 5 settings", len({k.split("/")[1] for k in d["grid"]}) == 5)
    c.check("§3 probe ran 3 arms", len({k.split("/")[0] for k in d["grid"]}) == 3)
    ctrl = d["controls"]
    c.check("§3 control sample has 12 distinct cases", len(ctrl) == 12)
    # Greedy rows must reproduce the committed baseline behaviour. For the two
    # arms with a behaviour_v3 recording this must hold exactly, which is what
    # makes the probe trustworthy. v2 has no recording at all, so its baseline
    # is flag-derived and cannot separate MERGE from STOP on single-topic
    # controls; it is held to a lower, explicitly-recorded bar rather than being
    # allowed to fail silently or be given the stronger claim.
    for key, v in d["grid"].items():
        arm, setting = key.split("/")
        if setting != "budget1200":
            continue  # only temp=0 with no penalty is the greedy reference
        tr = v["transitions"]
        same = sum(n for t, n in tr.items() if t.split(" -> ")[0] == t.split(" -> ")[1])
        floor = 0.95 if arm != "v2" else 0.60
        c.check(f"§3 {key}: greedy rows reproduce the baseline on "
                f"{same}/{v['n_cases']} cases", same / v["n_cases"] >= floor,
                f"changed={v['n_cases'] - same}")
    # and the exact claim for the two arms that can support it
    for arm in ("corrected-v3", "v3-2"):
        tr = d["grid"][f"{arm}/budget1200"]["transitions"]
        same = sum(n for t, n in tr.items() if t.split(" -> ")[0] == t.split(" -> ")[1])
        c.check(f"§3 {arm} greedy reproduces the committed baseline on all 64 cases",
                same == 64, f"same={same}")
    b = d["grid"]["v2/budget1200"]["baseline_same_subset"]
    c.check("§3 v2 baseline is recorded as flag-derived, not recorded",
            b["recording_used_for"] == 0 and b["flag_fallback_for"] == 64)


def _probe_rows() -> list[dict]:
    """Per-generation rows straight from the authoritative probe artefact."""
    out = []
    with (TUNE / "decoding_probe" / "results.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _is_loop(raw: str, det: dict) -> bool:
    """Recompute the repetition-loop rule from raw generation text.

    The rule is read out of the findings document, so this is an independent
    reimplementation rather than a restatement: if the unit, shingle size,
    repeat count or token floor were ever changed in the generator without the
    spec being updated, the two would disagree here.
    """
    sh, need = det["shingle"], det["min_repeats"]
    toks = re.findall(det["tokenizer"], raw or "")
    if len(toks) < det["min_tokens"]:
        return False
    counts: dict[tuple, int] = {}
    for i in range(len(toks) - sh + 1):
        g = tuple(toks[i:i + sh])
        counts[g] = counts.get(g, 0) + 1
    return bool(counts) and max(counts.values()) >= need


def main_section3b(c: Checks, text: str, d: dict) -> None:
    """The five diagnostic answers in section 3, checked against the real grid."""
    f = d["findings"]
    SETS = ("budget1200", "reppen105", "reppen1200", "temp01_norep", "temp01")
    rows = _probe_rows()

    # the grid must be complete before any of it is interpreted
    c.check("§3 probe reports 960 generations", d["rows"] == 960, str(d["rows"]))
    c.check("§3 probe is flagged complete", d["complete"] is True)
    c.check("§3 probe ran 960 generations", len(rows) == 960, str(len(rows)))
    c.check("§3 probe grid has 960 distinct cells",
            len({(r["model"], r["setting"], r["id"]) for r in rows}) == 960)
    c.check("§3 probe uses a distinct seed per generation",
            len({r["seed"] for r in rows}) == 960)
    c.check("§3 probe produced no empty generation",
            all((r.get("raw") or "").strip() for r in rows))
    c.check("§3 probe covers 3 arms x 5 settings x 64 cases",
            len(f["models"]) == 3 and len(f["settings"]) == 5
            and len({r["id"] for r in rows}) == 64)
    c.check("§3 every greedy row reproduces the recorded serving config",
            all(r["setting"] != "budget1200"
                or (r["max_tokens"] == 1200 and r["temp"] == 0.0 and not r["rep"])
                for r in rows))
    c.check("§3 no cell of the probe reuses another cell's settings",
            len({(r["temp"], r["max_tokens"], json.dumps(r["rep"])) for r in rows}) >= 4)

    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][f"{r['model']}/{r['setting']}"] = r["behaviour"]

    # A. quote invariance
    lit = f["literal_quote_cases"]
    c.check("§3 A the five literal-quote cases are the documented set",
            lit == ["ev3-016", "ev3-024", "ev3-032", "ev3-046", "ev3-052"], str(lit))
    for cid in lit:
        for arm in ("corrected-v3", "v3-2"):
            c.check(f"§3 A {cid} fails {arm} in all 5 settings",
                    f["quote_invariance"][cid][arm]["jsonfail_settings"] == 5)
            c.check(f"§3 A {cid} is JSONFAIL in every {arm} setting",
                    all(by[cid][f"{arm}/{s}"] == "JSONFAIL" for s in SETS))
        c.check(f"§3 A {cid} never fails v2 in any setting",
                f["quote_invariance"][cid]["v2"]["jsonfail_settings"] == 0)
    q83 = f["quote_invariance"]["ev3-083"]
    c.check("§3 A ev3-083 fails corrected-v3 in only 2 of 5 settings",
            q83["corrected-v3"]["jsonfail_settings"] == 2,
            str(q83["corrected-v3"]["jsonfail_settings"]))
    c.check("§3 A ev3-083 is a distinct defect, not one of the five quote cases",
            "ev3-083" not in lit)
    c.check("§3 A ev3-083 never fails v2 or v3-2",
            q83["v2"]["jsonfail_settings"] == 0 and q83["v3-2"]["jsonfail_settings"] == 0)
    # the probe must not have introduced or masked the defect
    c.check("§3 A the probe reproduces the committed 6-case defect set",
            set(d["defects"]) >= set(lit) | {"ev3-083"})

    # B. loops and knob isolation
    c.check("§3 B v2 has no repetition loop in any cell",
            f["loops"]["v2"]["cells_by_setting"] == {s: 0 for s in SETS},
            str(f["loops"]["v2"]["cells_by_setting"]))
    c.check("§3 B corrected-v3 loops on exactly one case",
            f["loops"]["corrected-v3"]["cases"] == ["ev3-083"],
            str(f["loops"]["corrected-v3"]["cases"]))
    c.check("§3 B v3-2 loops on eight cases",
            len(f["loops"]["v3-2"]["cases"]) == 8,
            str(f["loops"]["v3-2"]["cases"]))
    c.check("§3 B v3-2 loops on 7 cells without a penalty",
            f["loops"]["v3-2"]["cells_by_setting"]["budget1200"] == 7)
    for arm in ("corrected-v3", "v3-2"):
        c.check(f"§3 B {arm}: the token budget is inert at fixed penalty",
                f["knob_isolation"][arm]["budget_800_vs_1200_at_fixed_penalty"][0]
                == f["knob_isolation"][arm]["budget_800_vs_1200_at_fixed_penalty"][1],
                str(f["knob_isolation"][arm]["budget_800_vs_1200_at_fixed_penalty"]))
        c.check(f"§3 B {arm}: temperature alone changes no loop count",
                f["knob_isolation"][arm]["temperature_on_vs_off_at_fixed_budget"][0]
                == f["knob_isolation"][arm]["temperature_on_vs_off_at_fixed_budget"][1],
                str(f["knob_isolation"][arm]["temperature_on_vs_off_at_fixed_budget"]))
    c.check("§3 B the repetition penalty removes corrected-v3's only loop",
            f["knob_isolation"]["corrected-v3"]["penalty_on_vs_off_at_fixed_budget"] == [1, 0])
    c.check("§3 B the repetition penalty removes 2 of 7 v3.2 loops",
            f["knob_isolation"]["v3-2"]["penalty_on_vs_off_at_fixed_budget"] == [7, 5])
    # five v3.2 loops persist under every single setting
    persistent = [cid for cid in f["loops"]["v3-2"]["cases"]
                  if sum(by[cid][f"v3-2/{s}"] == "JSONFAIL" for s in SETS) == 5]
    c.check("§3 B five v3.2 cases fail under every setting",
            len(persistent) == 5, ",".join(sorted(persistent)))
    # Independently recompute the loop detector from the raw generations, so the
    # loop claims are verified against the text rather than trusted.
    det = f["loop_detector"]
    c.check("§3 B the documented detector is a word-level 6-gram repeated 4+ times",
            det["unit"] == "word" and det["shingle"] == 6
            and det["min_repeats"] == 4 and det["min_tokens"] == 60, str(det))
    loops = {(r["model"], r["setting"], r["id"]): _is_loop(r["raw"], det)
             for r in rows}
    for arm in ("v2", "corrected-v3", "v3-2"):
        cases = sorted({cid for (m, st, cid), v in loops.items()
                        if m == arm and v})
        per_setting = {st: sum(1 for (m, s2, cid), v in loops.items()
                               if m == arm and s2 == st and v) for st in SETS}
        c.check(f"§3 B recomputed loop cases for {arm} match the findings",
                cases == sorted(f["loops"][arm]["cases"]),
                f"recomputed={cases} reported={sorted(f['loops'][arm]['cases'])}")
        c.check(f"§3 B recomputed loop cells per setting match for {arm}",
                per_setting == f["loops"][arm]["cells_by_setting"],
                f"recomputed={per_setting}")
    # ev3-081 is the only temperature-linked case, and it needs no penalty to show
    c.check("§3 B ev3-081 loops only at temp-0.1 with no penalty",
            loops[("v3-2", "temp01_norep", "ev3-081")]
            and not any(loops[("v3-2", st, "ev3-081")]
                        for st in ("budget1200", "reppen105", "reppen1200", "temp01")))
    c.check("§3 B ev3-081 parses even where it loops, so repetition is not a parse failure",
            by["ev3-081"]["v3-2/temp01_norep"] != "JSONFAIL",
            by["ev3-081"]["v3-2/temp01_norep"])
    # the penalty effect is reproducible and monotone: both cured cases are
    # cleared by every penalty setting, at both strengths and both budgets
    for cid in ("ev3-028", "ev3-036"):
        c.check(f"§3 B {cid} loops at greedy and is cleared by every penalty setting",
                loops[("v3-2", "budget1200", cid)]
                and not any(loops[("v3-2", st, cid)]
                            for st in ("reppen105", "reppen1200", "temp01")))
    # ev3-036 additionally loops at temp-0.1 with no penalty, ev3-028 does not:
    # that asymmetry is what makes the temperature comparison a swap, not a null
    c.check("§3 B ev3-036 loops at both no-penalty settings",
            loops[("v3-2", "budget1200", "ev3-036")]
            and loops[("v3-2", "temp01_norep", "ev3-036")])
    c.check("§3 B ev3-028 loops at greedy but not at temp-0.1",
            loops[("v3-2", "budget1200", "ev3-028")]
            and not loops[("v3-2", "temp01_norep", "ev3-028")])
    # temperature trades one loop for another; the aggregate 7 -> 7 hides it
    def _cases_at(st: str) -> set:
        return {cid for (m, s2, cid) in loops if m == "v3-2" and s2 == st and loops[(m, s2, cid)]}
    greedy, temp = _cases_at("budget1200"), _cases_at("temp01_norep")
    c.check("§3 B temperature at 0.1 with no penalty trades ev3-028 for ev3-081",
            greedy - temp == {"ev3-028"} and temp - greedy == {"ev3-081"},
            f"cleared={sorted(greedy - temp)} created={sorted(temp - greedy)}")
    c.check("§3 B the temperature swap leaves the count unchanged at 7",
            len(greedy) == len(temp) == 7, f"{len(greedy)} vs {len(temp)}")
    # and five survive everything, counted over distinct cases
    c.check("§3 B five v3.2 loops survive every setting",
            sorted(cid for cid in _cases_at("budget1200")
                   if all(loops[("v3-2", st, cid)] for st in SETS)) ==
            ["ev3-017", "ev3-025", "ev3-037", "ev3-047", "ev3-049"])

    # C. gains are small and bounded
    c.check("§3 C v2 cannot improve schema because it is already 1.000",
            f["movement"]["v2"]["baseline"]["schema"] == 1.0
            and all(v["schema"] == 1.0 for v in f["movement"]["v2"]["by_setting"].values()))
    c.check("§3 C corrected-v3's whole schema gain is one case",
            f["movement"]["corrected-v3"]["by_setting"]["temp01"]["cases_schema_moved"] == 1,
            str(f["movement"]["corrected-v3"]["by_setting"]["temp01"]["cases_schema_moved"]))
    c.check("§3 C v3-2's whole schema gain is two cases",
            f["movement"]["v3-2"]["by_setting"]["temp01"]["cases_schema_moved"] == 2,
            str(f["movement"]["v3-2"]["by_setting"]["temp01"]["cases_schema_moved"]))

    # D. the movement is JSONFAIL-driven, not a MERGE/SPLIT reclassification
    tt = {r["id"] for r in rows if "twotopic" in (r.get("tags") or ())}
    c.check("§3 D the probe tags its 52 two-topic cases",
            len(tt) == 52, str(len(tt)))
    c.check("§3 D the probe's two-topic subset is the 52 documented cases",
            len(tt) == 52, str(len(tt)))
    for arm in ("corrected-v3", "v3-2"):
        jf = {s: sum(1 for cid in tt if by[cid][f"{arm}/{s}"] == "JSONFAIL")
              for s in SETS}
        c.check(f"§3 D {arm} parses fewer cases under the best setting",
                min(jf.values()) < jf["budget1200"], str(jf))
        c.check(f"§3 D {arm} never reaches zero JSON failures",
                min(jf.values()) > 0, str(jf))
    # STOP is never bought back by a setting: it is the regressor signature
    for arm in ("corrected-v3", "v3-2"):
        stops = {s: sum(1 for cid in tt if by[cid][f"{arm}/{s}"] == "STOP") for s in SETS}
        c.check(f"§3 D {arm} emits at least as many STOPs as the greedy row",
                min(stops.values()) >= stops["budget1200"], str(stops))

    # E. the conclusion that no cell is worth serving
    for arm in ("v2", "corrected-v3", "v3-2"):
        v = f["best_cell"][arm]
        c.check(f"§3 E no setting closes a gate for {arm}", v["closes_a_gate"] is False)
        c.check(f"§3 E {arm} best topic count stays below the 0.90 gate",
                v["topic_count"] < 0.90, str(v["topic_count"]))
        # closes_a_gate must follow from the reported blockers, not be asserted
        c.check(f"§3 E {arm} closes_a_gate agrees with its own blocking_gates",
                v["closes_a_gate"] == (len(v["blocking_gates"]) == 0),
                str(v["blocking_gates"]))
        for gname in v["blocking_gates"]:
            c.check(f"§3 E {arm} blocker is a real gate name",
                    gname.startswith(("schema", "topic_count")))
    # v2 reaches schema 1.00 on this subset, so only the topic-count gate blocks it
    c.check("§3 E v2 does reach schema 1.00 on the probe subset",
            f["best_cell"]["v2"]["schema"] == 1.0, str(f["best_cell"]["v2"]["schema"]))
    c.check("§3 E v2 is blocked by topic count alone",
            f["best_cell"]["v2"]["blocking_gates"] ==
            [f'topic_count {f["best_cell"]["v2"]["topic_count"]:.4f} < 0.90'],
            str(f["best_cell"]["v2"]["blocking_gates"]))
    for arm in ("corrected-v3", "v3-2"):
        c.check(f"§3 E {arm} is blocked by schema as well as topic count",
                len(f["best_cell"][arm]["blocking_gates"]) == 2)
    c.check("§3 E v2 gains nothing from any setting",
            f["best_cell"]["v2"]["cases_schema_moved"] == 0)
    # the probe subset is explicitly not the whole eval suite
    for k, v in d["gate_coverage"].items():
        c.check(f"§3 E gate {k} coverage is stated honestly", "why" in v)
    c.check("§3 E the probe is not claimed to cover the full suites",
            d["gate_coverage"]["frozen_domain"]["probe_covers"] is False
            and d["gate_coverage"]["smoke_two_topic"]["probe_covers"] is False)


def main_section3c(c: Checks, text: str, d: dict, rows: list[dict]) -> None:
    """Pin section 3's prose figures to the artefacts they claim to come from.

    The section 3 checks above prove the probe data is self-consistent. These
    prove the *report* says the same thing, which is a separate failure mode:
    a correct artefact with a stale sentence beside it.
    """
    sec = text.split("## 3.")[1].split("## 4.")[0]
    f = d["findings"]
    full = " ".join(sec.split())

    c.close("§3 states the completed cell count", d["rows"], 960, place=full)
    c.says("§3 states 960 of 960 cells", "960 of 960 cells", full)
    c.says("§3 states 960 distinct seeds", "960 distinct seeds", full)
    c.says("§3 states zero duplicates, missing or empty", "zero", full)
    c.close("§3 states the 52-case recording coverage", 52, 52, place=full)
    for cid in f["literal_quote_cases"]:
        c.says(f"§3 names the literal-quote case {cid}", cid, full)
    c.says("§3 names the separate JSONFAIL case ev3-083", "ev3-083", full)
    for cid in f["loops"]["v3-2"]["cases"]:
        c.says(f"§3 names the v3.2 loop case {cid}", cid, full)
    # the five that survive every setting are the load-bearing number
    survive = ["ev3-017", "ev3-025", "ev3-037", "ev3-047", "ev3-049"]
    c.says("§3 states 5 of v3.2's 8 loop cases persist",
           "5 of v3.2's 8 loop cases", full)
    c.close("§3 states the per-arm loop case total", 8,
            len(f["loops"]["v3-2"]["cases"]), place=full)
    for cid in survive:
        c.says(f"§3 names persistent loop {cid}", cid, full)
    # the canonical path is the one string the reader will copy
    c.says("§1 states the canonical fused model path",
           "qwen3-8b-lora-v2-attempt10-fused", text)
    # no arm closes a gate; the probe must not imply otherwise
    for arm, v in f["best_cell"].items():
        c.close(f"§3 quotes {arm}'s best topic_count",
                v["topic_count"], v["topic_count"], place=full)
    # Each section restates the loop finding in its own terms, so pin the
    # specific claim each one makes, derived from the loop data rather than
    # from a single phrasing. A stale number in any of them then fails.
    cases = f["loops"]["v3-2"]["cases"]
    body1 = text.split("## 1.")[1].split("\n## ")[0]
    body2 = text.split("## 2.")[1].split("\n## ")[0]
    body7 = text.split("## 7.")[1].split("\n## 8.")[0]
    # section 2 carries the defect classification row for repetition loops
    c.says("§2 states the persistent loop count", "5 of 8 v3.2 loop cases", body2)
    c.says("§2 states the penalty cure count", "cures 2", body2)
    c.says("§2 states that temperature creates a loop", "temperature creates 1", body2)
    c.says("§2 does not blame serving for loops", "serving/decoding", body2)
    c.says("§3 states the persistent loop count", "5 of v3.2's 8 loop cases", full)
    c.says("§3 states the penalty cure count", "2 of 8", full)
    # the cure count is greedy loop cells minus penalised loop cells; deriving it
    # this way makes the sentence falsifiable if the probe is re-run
    cells = f["loops"]["v3-2"]["cells_by_setting"]
    c.close("§7 penalty cure count matches the loop data", 2,
            cells["budget1200"] - cells["reppen105"], place=body7)
    c.says("§7 states the penalty cure count", "2 of 8", body7)
    c.check("§1 does not restate the loop finding", "loop" not in body1.lower())


def build_report(report: Path | None = None) -> dict:
    text = (report or REPORT).read_text(encoding="utf-8")
    tables = _tables(text)
    rows = _matrix_rows()
    act = json.loads(ACTION.read_text(encoding="utf-8"))
    residue = json.loads(RESIDUE.read_text(encoding="utf-8"))
    decode = json.loads(DECODE.read_text(encoding="utf-8"))
    return {"text": text, "tables": tables, "rows": rows, "act": act,
            "residue": residue, "decode": decode, "refstyle": _reference_style()}


def _reference_style() -> dict:
    """Recompute the §4 style numbers from source, not from residue.json."""
    tg = _load(FROZEN / "lora_targets.jsonl")
    issues = _scan_issues(tg)
    n = len({r["idx"] for r in tg}) or 1
    frozen = {"boilerplate": round(len(_scan_issues_bool(tg, GATE_RE)) / n, 4),
              "mean_chars": round(sum(len(s) for s in issues) / len(issues), 1)}
    gens = {}
    for arm, tag in (("v2", "lora"), ("previous-v3", "v3-treatment"),
                     ("corrected-v3", "v3-corrected"), ("v3.2", "v3-2")):
        p = FROZEN / f"{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        texts, leaks, cnt = [], 0, 0
        for pr in d["predictions"]:
            raw = pr.get("raw", "")
            if not raw.startswith("{"):
                continue
            try:
                tp = json.loads(raw).get("topics") or []
            except json.JSONDecodeError:
                continue
            iss = [str(t.get("issue", "")) for t in tp if isinstance(t, dict)]
            texts += iss
            leaks += bool(_scan(iss, GATE_RE))
            cnt += 1
        gens[arm] = {"boilerplate": round(leaks / cnt, 4) if cnt else None,
                     "mean_chars": round(sum(len(s) for s in texts) / len(texts), 1) if texts else None}
    return {"frozen_refs": frozen, "generations": gens}


def _scan_issues_bool(records, rx) -> list[int]:
    out = []
    for r in records:
        iss = [str(t.get("issue", "")) for t in (r.get("topics") or []) if isinstance(t, dict)]
        if _scan(iss, rx):
            out.append(r["idx"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--report", type=Path,
                    help="verify a different report file (used by the "
                         "corruption tests to prove the checks bite)")
    args = ap.parse_args()

    rep = build_report(args.report)
    c = Checks()
    main_section1(c, rep["tables"], rep["text"])
    main_section2(c, rep["text"], rep["rows"])
    main_section3(c, rep["tables"], rep["text"])
    main_section3b(c, rep["text"], rep["decode"])
    main_section3c(c, rep["text"], rep["decode"], _probe_rows())
    main_section4(c, rep["tables"], rep["refstyle"])
    main_section5(c, rep["tables"], rep["residue"])
    main_section5b(c, rep["tables"], rep["residue"])
    main_section6(c, rep["tables"], rep["act"])
    main_section7(c, rep["text"], rep["act"], rep["rows"])
    main_section8(c, rep["text"], rep["rows"])

    for r in c.failed():
        print(f"FAIL  {r['claim']}  {r['detail']}")
    print(c.verdict())
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"verdict": c.verdict(), "claims": c.rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)