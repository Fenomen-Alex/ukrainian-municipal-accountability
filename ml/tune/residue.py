"""Direct measurement of the R1/R2/R3 residue claims.

The suite's own ``boilerplate_leak_rate`` gate covers the R2 delivery-mode
pattern, but only its ``в`` form -- ``BOILER_SENT`` contains
``в\\s+телефонному\\s+режимі`` and not ``у\\s+телефонному\\s+режимі``, which is the
same euphony blind spot ``v3_changes`` documents and works around. It does not
cover the R3 generic closer at all. So neither R2 nor R3 can be confirmed from
the gate alone, and this module measures them from the raw generations with a
detector that covers both euphonic forms.

Two detectors are therefore reported side by side:

``gate_patterns``   exactly the patterns ``boilerplate_leak`` uses, so the number
                    here reconciles with the committed metric.
``full_patterns``   the same set plus the ``у`` forms and the R3 closer.

A disagreement between the two is itself the finding: it bounds how much residue
the gate cannot see.

Usage::

    .venv/bin/python -m ml.tune.residue
    .venv/bin/python -m ml.tune.residue --json-out FILE
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ml.tune.build_eval_v3 import BOILER_SENT
from ml.tune.run_eval_v3 import parse_payload

ROOT = Path(__file__).resolve().parents[2]
TUNE = ROOT / "ml/data/tune"
FROZEN_FILE = {"v2": "lora", "previous-v3": "v3-treatment",
              "corrected-v3": "v3-corrected", "v3.2": "v3-2"}

GATE_RE = re.compile("|".join(BOILER_SENT), re.IGNORECASE)

#: R2 delivery mode in both euphonic forms. Ukrainian alternates в/у by the
#: preceding sound; v3_changes counts 584 "в" against 127 "у" in this corpus.
R2_EXTRA = (r"у\s+телефонному\s+режимі", r"у\s+письмовому\s+режимі")
#: R3 generic closer: "Прохання вжити заходи.", "Просить вжити заходів."
R3 = (r"(прохання|просить|просимо|прошу|проханням)\s+(терміново|негайно)?\s*"
      r"вжити\s+(необхідні|необхідних|відповідні|належні|належних)?\s*"
      r"заход\w*\s*(реагування)?\s*[.!]?$")

R2_RE = re.compile("|".join((r"[ву]\s+у?\s*(?:телефонном\w*|письмов\w*)\s+режим\w*",)),
                   re.IGNORECASE)
R3_RE = re.compile(R3, re.IGNORECASE)

BEHAVIOUR_FILE = {"previous-v3": "v3-prev-recheck",
                 "corrected-v3": "v3-corrected-recheck", "v3.2": "v3-2"}

ARMS = [("v2", "lora"), ("previous-v3", "v3-treatment"),
        ("corrected-v3", "v3-corrected"), ("v3.2", "v3-2")]


def _issue_texts(raw: str) -> list[str]:
    return [str(t.get("issue", "")) for t in (parse_payload(raw).topics or [])
            if isinstance(t, dict)]


def _scan(texts: list[str], rx: re.Pattern) -> list[str]:
    """Sentences carrying the pattern, at sentence granularity like the gate."""
    out = []
    split = re.compile(r"(?<=[.!?])\s+")
    for issue in texts:
        for sent in split.split(issue):
            if sent and rx.search(sent):
                out.append(sent.strip())
    return out


def corpus_residue(path: Path, limit: int | None = None) -> dict:
    n = leak = r2 = r3 = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if limit and i >= limit:
            break
        rec = json.loads(line)
        try:
            payload = json.loads(rec["messages"][-1]["content"])
        except (KeyError, json.JSONDecodeError):
            continue
        n += 1
        texts = [str(t.get("issue", "")) for t in payload.get("topics", [])]
        if _scan(texts, GATE_RE):
            leak += 1
        if _scan(texts, R2_RE):
            r2 += 1
        if _scan(texts, R3_RE):
            r3 += 1
    return {"n": n, "gate_leak": leak, "r2_delivery_mode": r2, "r3_generic_closer": r3,
            "gate_rate": round(leak / n, 4) if n else None,
            "r2_rate": round(r2 / n, 4) if n else None,
            "r3_rate": round(r3 / n, 4) if n else None}


def frozen_residue(arm: str, tag: str) -> dict:
    run_p = TUNE / "eval" / f"{tag}.json"
    if not run_p.exists():
        return {}
    preds = json.loads(run_p.read_text(encoding="utf-8"))["predictions"]
    gate = r2 = r3 = 0
    for p in preds:
        texts = _issue_texts(p["raw"])
        if _scan(texts, GATE_RE):
            gate += 1
        if _scan(texts, R2_RE):
            r2 += 1
        if _scan(texts, R3_RE):
            r3 += 1
    n = len(preds)
    return {"n": n, "gate_leak": gate, "r2_delivery_mode": r2,
            "r3_generic_closer": r3,
            "gate_rate": round(gate / n, 4), "r2_rate": round(r2 / n, 4),
            "r3_rate": round(r3 / n, 4)}


def eval_v3_residue(arm: str) -> dict:
    p = TUNE / "eval_v3/behaviour" / f"{arm}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    gate = r2 = r3 = 0
    for c in d["cases"]:
        texts = _issue_texts(c.get("raw", ""))
        if _scan(texts, GATE_RE):
            gate += 1
        if _scan(texts, R2_RE):
            r2 += 1
        if _scan(texts, R3_RE):
            r3 += 1
    n = len(d["cases"])
    return {"n": n, "gate_leak": gate, "r2_delivery_mode": r2,
            "r3_generic_closer": r3, "gate_rate": round(gate / n, 4),
            "r2_rate": round(r2 / n, 4), "r3_rate": round(r3 / n, 4)}


#: Literal empty-double-quote pair. Not Ukrainian quoting -- it is an artefact of
#: whatever normalised the source text. It is *not* JSON-safe: copying it into a
#: string value terminates the value early, so any output containing it fails to
#: parse. This is the whole mechanism behind the v3 arms' quote JSON failures.
QUOTE_ARTEFACT = re.compile(r"\"\"")


def quote_artefact_corpus(path: Path) -> dict:
    """How much literal "" the training targets teach the model to copy."""
    spans: Counter[str] = Counter()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        for m in row["messages"]:
            if m.get("role") == "assistant":
                spans.update(x.group(0) for x in QUOTE_ARTEFACT.finditer(m["content"]))
    return {"rows": n, "empty_quote_occurrences": sum(spans.values()),
            "examples_with_one": sum(1 for v in spans.values() if v),
            "distinct_delimiters": dict(spans.most_common(5))}


def quote_artefact_eval_v3(arms: list[str]) -> dict:
    """Do the eval_v3 *inputs* carry the same artefact, and who survives it?

    The five cases whose input contains "" are the whole story: v2 parses all
    five, and both v3 arms parse none of them.
    """
    cases = [json.loads(l) for l in
             (TUNE / "eval_v3/cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    affected = [c["id"] for c in cases if QUOTE_ARTEFACT.search(c["text"])]
    out: dict = {"n_cases": len(cases), "affected_ids": affected, "by_arm": {}}
    for arm in arms:
        f = TUNE / f"eval_v3/results/{arm}.json"
        if not f.exists():
            continue
        recs = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["per_case"]}
        out["by_arm"][arm] = {
            "json_ok": sum(1 for i in affected if recs[i]["json_ok"]),
            "json_fail": [i for i in affected if not recs[i]["json_ok"]],
        }
    return out


def build() -> dict:
    return {
        "quote_artefact": {
            "detector": QUOTE_ARTEFACT.pattern,
            "corpus": {
                "v2 train": quote_artefact_corpus(TUNE / "v2/train.jsonl"),
                "v3.2 train": quote_artefact_corpus(TUNE / "v3/treatment_v3_2/train.jsonl"),
            },
            "eval_v3": quote_artefact_eval_v3(["v2", "v3-corrected", "v3-2"]),
        },
        "detectors": {"gate": BOILER_SENT, "r2_extra": R2_EXTRA, "r3": R3},
        "corpus": {
            "v2 train": corpus_residue(TUNE / "v2/train.jsonl"),
            "v3.2 train": corpus_residue(TUNE / "v3/treatment_v3_2/train.jsonl"),
        },
        "frozen_329": {a: frozen_residue(a, FROZEN_FILE.get(a, a)) for a, _ in ARMS},
        "behaviour_52": {a: eval_v3_residue(f) for a, f in BEHAVIOUR_FILE.items()},
    }


def to_markdown(r: dict) -> dict:
    L = ["# R1/R2/R3 residue measurement\n",
         "\n## Training corpora (targets)\n",
         "| corpus | rows | gate boilerplate | R2 delivery mode | R3 generic closer |",
         "|---|---|---|---|---|"]
    for k, v in r["corpus"].items():
        L.append(f"| {k} | {v['n']} | {v['gate_rate']:.4f} | {v['r2_rate']:.4f} | {v['r3_rate']:.4f} |")
    L.append("\n## Model generations, frozen 329\n")
    L.append("| arm | n | gate boilerplate | R2 | R3 |")
    L.append("|---|---|---|---|---|")
    for k, v in r["frozen_329"].items():
        if v:
            L.append(f"| {k} | {v['n']} | {v['gate_rate']:.4f} | {v['r2_rate']:.4f} | {v['r3_rate']:.4f} |")
    L.append("\n## Model generations, 52 two-topic cases\n")
    L.append("| arm | n | gate boilerplate | R2 | R3 |")
    L.append("|---|---|---|---|---|")
    for k, v in r["behaviour_52"].items():
        if v:
            L.append(f"| {k} | {v['n']} | {v['gate_rate']:.4f} | {v['r2_rate']:.4f} | {v['r3_rate']:.4f} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()
    r = build()
    if a.json_out:
        a.json_out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {a.json_out}")
    print(to_markdown(r))


if __name__ == "__main__":
    main()