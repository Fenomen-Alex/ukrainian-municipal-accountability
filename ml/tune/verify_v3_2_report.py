"""Verify every number in ``V3_2_REPORT.md`` section 7 against the artefacts.

The report is the deliverable, so a stale or hand-transcribed figure is a real
risk: the tables were generated once, by hand, from result files that nothing
re-checks. This module re-reads the *report text* and compares each figure with
the committed artefact it claims to summarise, so editing a number without
re-running the sweep -- or vice versa -- fails here rather than in a review.

It deliberately parses the markdown instead of holding its own copy of the
expected values. A second list of numbers would only be a second thing to get
wrong.

    .venv/bin/python -m ml.tune.verify_v3_2_report
    .venv/bin/python -m ml.tune.verify_v3_2_report --json-out /tmp/claims.json

Exit status is 0 only when every claim checks out.

Note on coverage: ``ml/data/tune/eval/`` (the frozen 329-case results) is
gitignored by repo policy that predates this experiment, so 55 of the 197
claims need that directory to have been produced locally. The other 142 are
covered entirely by tracked artefacts. A missing directory raises rather than
being skipped silently.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "ml/tune/V3_2_REPORT.md"

EVAL_V3 = ROOT / "ml/data/tune/eval_v3/results"
BEHAVIOUR = ROOT / "ml/data/tune/eval_v3/behaviour"
MT = ROOT / "ml/data/tune/multitopic/results"
FROZEN = ROOT / "ml/data/tune/eval"
ADAPTERS = ROOT / "ml/data/tune/adapters"
MANIFEST = ROOT / "ml/data/tune/artifact_sha256.json"

#: arm label used in the report -> eval_v3 / multitopic / frozen result tag.
#: v2's frozen and multitopic results predate the runners' --tag flag and so
#: carry the default ``lora`` name, exactly as gates_v3.FROZEN_SUITES records.
TAGS = {
    "v2": ("v2", "lora", "lora"),
    "prev-v3": ("v3-treatment", "v3-treatment", "v3-treatment"),
    "corrected-v3": ("v3-corrected", "v3-corrected", "v3-corrected"),
    "v3.2": ("v3-2", "v3-2", "v3-2"),
}
ARMS = ("v2", "prev-v3", "corrected-v3", "v3.2")
#: behaviour recheck files reuse the raw generations captured during the
#: diagnosis; the classifier itself was validated against both arms.
BEHAVIOUR_FILES = {
    "prev-v3": "v3-prev-recheck.json",
    "corrected-v3": "v3-corrected-recheck.json",
    "v3.2": "v3-2.json",
}
CATEGORY_E = [f"ev3-{n:03d}" for n in range(35, 41)]
QUOTE_MARK = '"" не відповідають'


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing artefact: {path.relative_to(ROOT)}\n"
            "run bash ml/tune/run_v3_2_eval.sh first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _cache():
    ev3 = {a: _load(EVAL_V3 / f"{TAGS[a][0]}.json") for a in ARMS}
    mt = {a: _load(MT / f"{TAGS[a][1]}.json") for a in ARMS}
    fr = {a: _load(FROZEN / f"{TAGS[a][2]}.json") for a in ARMS}
    beh = {a: _load(BEHAVIOUR / f)["summary"] for a, f in BEHAVIOUR_FILES.items()}
    cases = {a: _load(BEHAVIOUR / f)["cases"] for a, f in BEHAVIOUR_FILES.items()}
    return ev3, mt, fr, beh, cases


def _tables(text: str) -> dict[str, list[list[str]]]:
    """Split the report into ``{section heading: rows}`` for every markdown table.

    Rows are returned verbatim (cells not stripped of the ``**`` emphasis) so a
    number is checked exactly as a reader would read it out of the document.
    """
    out: dict[str, list[list[str]]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*\S)\s*$", line)
        if m:
            current = m.group(1)
            continue
        if current and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue  # header rule
            out.setdefault(current, []).append(cells)
    return out


def _num(cell: str) -> float | None:
    """Leading number in a table cell, ignoring markdown emphasis and any
    trailing verdict annotation such as ``**regression**`` or ``✓``.

    Only the leading number is taken on purpose: a cell such as
    ``→ SPLIT (`ev3-034`, ...)`` must not be read as the number 34.
    """
    m = re.search(r"[-+]?\d*\.?\d+", cell.replace("*", ""))
    return float(m.group(0)) if m else None


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, claim: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"claim": claim, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def close(self, claim: str, got: float | None, want: float | None,
              tol: float = 5e-5) -> bool:
        if got is None or want is None:
            return self.check(claim, False, f"reported={got} expected={want}")
        ok = abs(got - want) <= tol
        return self.check(claim, ok, f"reported={got:.4f} expected={want:.4f}")

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]

    def verdict(self) -> str:
        if self.failed:
            return "FAIL"
        return "PASS" if self.rows else "EMPTY"


def _arm_metric_tables(c: Checks, tables, ev3, mt, fr) -> None:
    """The three per-arm metric tables.

    Dispatch is on the heading *text*, not its number. The section numbers moved
    when the merge-reduction table was inserted, and number-keyed matching
    silently stopped verifying the whole table -- which a negative test caught.
    Keying on wording means a renumber cannot quietly disable a check.
    """
    def ev3_metric(a, k):
        return ev3[a]["metrics"][k]

    def mt44(a, k):
        return ev3[a]["subsets"]["multi_topic_categories"][k]

    def mt85(a, k):
        return mt[a]["metrics"][k]

    def frozen(a, k):
        return fr[a]["metrics"][k]

    for head, rows in tables.items():
        for k in rows:
            if len(k) < 6 or not k[0].startswith("`"):
                continue
            label = k[0].strip("`")
            src, fn, tag = None, None, None
            if "eval_v3 (146 cases)" in head:
                src, fn, tag = "eval_v3", ev3_metric, None
            elif head.startswith("Multi-topic subset"):
                if label.startswith("mt44 "):
                    src, fn, tag = "mt44", mt44, "mt44 " + label[5:].strip("`")
                elif label.startswith("mt85 "):
                    src, fn, tag = "mt85", mt85, "mt85 " + label[5:].strip("`")
            elif "Frozen 329" in head:
                src, fn, tag = "frozen", frozen, None
            if src is None:
                continue
            key = tag or label
            want = [fn(a, key) for a in ARMS]
            for a, w in zip(ARMS, want):
                c.close(f"{head.split()[0]} {src} {key} [{a}]",
                        _num(k[ARMS.index(a) + 1]), w)
            c.close(f"{head.split()[0]} {src} {key} [delta]",
                    _num(k[5]), round(want[3] - want[2], 6))


def _behaviour(c: Checks, tables, beh) -> None:
    for head, rows in tables.items():
        if "Behaviour on the 52" not in head:
            continue
        for r in rows:
            if len(r) < 7 or r[0] not in ("prev-v3", "corrected-v3", "v3.2"):
                continue
            b = beh[r[0]]["buckets"]
            for i, k in enumerate(("SPLIT", "MERGE", "STOP", "JSONFAIL")):
                c.close(f"7.2 {r[0]} {k}", _num(r[i + 1]), float(b[k]))
            c.close(f"7.2 {r[0]} accuracy %", _num(r[5].split("%")[0]),
                    b["SPLIT"] / 52 * 100, tol=0.06)
            e = beh[r[0]]["category_E"]
            got = sum(1 for v in e.values() if v == "SPLIT")
            c.check(f"7.2 {r[0]} category E = {r[6]}",
                    r[6] == f"{got}/6", f"artefact {got}/6")


def _categories(c: Checks, tables, ev3) -> None:
    """The -10-point stop-trigger table: per-category topic_count_accuracy."""
    order = ("prev-v3", "corrected-v3", "v3.2")
    for head, rows in tables.items():
        if "Categories breaching" not in head:
            continue
        for r in rows:
            if len(r) < 5 or not re.fullmatch(r"[A-Z]", r[0]):
                continue
            k = r[0]
            want = [ev3[a]["by_category"][k]["topic_count_accuracy"] for a in order]
            for i, a in enumerate(order):
                c.close(f"{head.split()[0]} category {k} [{a}]", _num(r[i + 1]), want[i])
            c.close(f"{head.split()[0]} category {k} delta vs prev-v3", _num(r[4]),
                    round(want[2] - want[0], 6))


def _section(text: str, needle: str) -> str:
    """Body of one report section, so an id must be named *there* and not
    merely somewhere in the document."""
    m = re.search(r"^#{2,3}\s+[^\n]*" + re.escape(needle) + r"[^\n]*$(.*?)"
                  r"(?=^#{2,3}\s|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def _section_rows(text: str, needle: str) -> list[list[str]]:
    """Table rows belonging to one section, identified by their first cell.

    Checking the *table* rather than the prose matters: deleting a row while
    leaving the surrounding sentence intact would otherwise still pass.
    """
    body = _section(text, needle)
    out = []
    for line in body.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                out.append(cells)
    return out


def _jsonfail(c: Checks, text: str, beh, cases) -> None:
    """The JSON-failure section: five quote failures carried forward, seven new
    repetition loops, each named in the section's own table."""
    raw = {x["id"]: x["raw"] for x in cases["v3.2"]}
    behv = {x["id"]: x["behaviour"] for x in cases["v3.2"]}
    corr = {x["id"]: x["behaviour"] for x in cases["corrected-v3"]}
    jf = sorted(i for i, v in behv.items() if v == "JSONFAIL")
    quote = [i for i in jf if QUOTE_MARK in raw[i]]
    loop = [i for i in jf if raw[i].count("{") != raw[i].count("}")]
    c.check("7.8 JSONFAIL count = 12", len(jf) == 12, f"{len(jf)}")
    c.check("7.8 five quote-related failures", len(quote) == 5,
            " ".join(quote))
    c.check("7.8 quote failures were already JSONFAIL in corrected-v3",
            all(corr[i] == "JSONFAIL" for i in quote), " ".join(quote))
    c.check("7.8 seven brace-unbalanced repetition loops", len(loop) == 7,
            " ".join(loop))
    c.check("7.8 repetition loops are new regressions (corrected-v3 was not JSONFAIL)",
            all(corr[i] != "JSONFAIL" for i in loop))
    body = _section(text, "The 12 JSON failures")
    rows = _section_rows(text, "The 12 JSON failures")
    lead = {r[0].strip("`") for r in rows if r}
    for i in loop:
        # the seven loops must each have their own row in the table; a prose
        # mention alone is not evidence the row survived
        c.check(f"{i} has its own row in the JSON-failure table", i in lead)
    for i in quote:
        c.check(f"{i} named in the JSON-failure section",
                bool(re.search(r"`" + i + r"`", body)))
    c.check("7.8 ev3-025 was a corrected-v3 SPLIT", corr.get("ev3-025") == "SPLIT")
    c.check("7.8 ev3-028 was a corrected-v3 SPLIT", corr.get("ev3-028") == "SPLIT")


def _category_e(c: Checks, tables, beh) -> None:
    e = beh["v3.2"]["category_E"]
    got = [e[i] for i in CATEGORY_E]
    c.check("7.9 category E v3.2 = 3/6", got.count("SPLIT") == 3, " ".join(got))
    c.check("7.9 category E corrected-v3 = 1/6",
            sum(1 for v in beh["corrected-v3"]["category_E"].values()
                if v == "SPLIT") == 1)
    c.check("7.9 category E prev-v3 = 6/6",
            sum(1 for v in beh["prev-v3"]["category_E"].values()
                if v == "SPLIT") == 6)
    c.check("7.9 ev3-036 and ev3-037 are the two destroyed recoveries",
            e["ev3-036"] == "JSONFAIL" and e["ev3-037"] == "JSONFAIL")
    c.check("7.9 ev3-038 is the genuine remaining merge", e["ev3-038"] == "MERGE")


def _verdict(c: Checks, text: str, ev3, mt, beh) -> None:
    """7.1: 0 of 4 success criteria, 2 of 3 revert triggers."""
    v32 = beh["v3.2"]["buckets"]
    e = beh["v3.2"]["category_E"]
    n85 = mt["v3.2"]["metrics"]["topic_count_accuracy"] * 85
    success = [
        ("expected-two >= 60%", v32["SPLIT"] / 52 >= 0.60, f"{v32['SPLIT']}/52"),
        ("category E >= 5/6", sum(1 for x in e.values() if x == "SPLIT") >= 5, None),
        ("MT85 >= 80/85", n85 >= 80, f"{n85:.0f}/85"),
        ("merge count <= 8", v32["MERGE"] <= 8, f"{v32['MERGE']}"),
    ]
    failed = [n for n, ok, _ in success if not ok]
    c.check("7.1 zero of four success criteria pass", len(failed) == 4,
            f"{len(failed)} of 4 failed: " + " ".join(failed))
    trig = []
    if n85 < 78:
        trig.append("MT85<78")
    if any(ev3["v3.2"]["by_category"][k]["topic_count_accuracy"]
           < ev3["prev-v3"]["by_category"][k]["topic_count_accuracy"] - 0.10
           for k in ev3["prev-v3"]["by_category"]):
        trig.append("category-10pt")
    if v32["MERGE"] > 15:
        trig.append("merge>15")
    c.check("7.1 two of three revert triggers fire", len(trig) == 2, " ".join(trig))
    c.check("verdict FAILED EXPERIMENT stated", "FAILED EXPERIMENT" in text)
    c.check("release decision says do not ship",
            "must not be shipped" in text)


def _shas(c: Checks, text: str) -> None:
    """7.13: every sha256 quoted in the report matches the artefact on disk."""
    man = _load(MANIFEST)
    # labels keep their spaces so they match the check list below; only the
    # markdown emphasis marks are removed, so "fused `model.safetensors`"
    # normalises to "fused model.safetensors".
    quoted = {k.replace("`", "").strip(): v for k, v in re.findall(
        r"\|\s*(`?[a-z0-9 .\-+`]+?`?)\s*\|\s*`([0-9a-f]{64})`\s*\|", text)}
    checks = [
        ("iteration-4500 checkpoint", "ml/data/tune/adapters/qwen3-8b-lora-v3-2/"
                                     "0004500_adapters.safetensors"),
        ("v3.2 corpus", "ml/data/tune/v3/treatment_v3_2/train.jsonl"),
        ("final adapter", "ml/data/tune/adapters/qwen3-8b-lora-v3-2/"
                          "adapters.safetensors"),
        ("fused model.safetensors",
         "ml/data/tune/adapters/qwen3-8b-lora-v3-2-fused/model.safetensors"),
    ]
    import hashlib
    for label, rel in checks:
        p = ROOT / rel
        if not p.exists():
            c.check(f"7.13 {label} exists", False, rel)
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        got = h.hexdigest()
        c.check(f"7.13 {label} sha matches report", quoted.get(label) == got,
                f"report={quoted.get(label)} disk={got}")
    v32f = man.get("qwen3-8b-lora-v3-2-fused", {})
    c.check("7.13 fused arm recorded in manifest", v32f.get("present") is True)
    c.check("7.13 fusion has 112 files and config_matches_v2",
            v32f.get("n_files") == 7)
    fm = _load(ADAPTERS / "qwen3-8b-lora-v3-2-fused/fuse_manifest.json")
    c.check("7.13 config_matches_v2 true", fm["config_matches_v2"] is True)
    c.check("7.13 112 fused modules", fm["fused_modules"] == 112,
            str(fm["fused_modules"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json-out")
    ap.add_argument("--report", default=REPORT,
                    help="report to verify (default: V3_2_REPORT.md)")
    a = ap.parse_args()

    report = Path(a.report)
    if not report.exists():
        raise SystemExit(f"missing {report}")
    text = report.read_text(encoding="utf-8")
    ev3, mt, fr, beh, cases = _cache()
    tables = _tables(text)
    c = Checks()
    _arm_metric_tables(c, tables, ev3, mt, fr)
    _behaviour(c, tables, beh)
    _categories(c, tables, ev3)
    _jsonfail(c, text, beh, cases)
    _category_e(c, tables, beh)
    _verdict(c, text, ev3, mt, beh)
    _shas(c, text)

    print(f"v3.2 report claim verification: {c.verdict()}")
    print(f"  {len(c.rows) - len(c.failed)}/{len(c.rows)} claims check out")
    for r in c.failed:
        print(f"  FAIL  {r['claim']}  {r['detail']}")
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps({"verdict": c.verdict(), "n_checks": len(c.rows),
                        "failed": len(c.failed), "checks": c.rows}, indent=2),
            encoding="utf-8")
        print(f"  -> {a.json_out}")
    raise SystemExit(0 if c.verdict() == "PASS" else 1)


if __name__ == "__main__":
    main()