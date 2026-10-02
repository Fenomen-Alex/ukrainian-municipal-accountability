"""The report-claim verifier must be able to fail.

`verify_v3_2_report.py` exists so a stale or hand-edited figure in
`V3_2_REPORT.md` cannot ship unnoticed. That guarantee is worthless if the
verifier silently matches nothing -- which is exactly what happened once: the
verifier keyed its table dispatch on section *numbers*, a later section was
inserted, and 115 figure checks quietly stopped running while the tool still
reported PASS.

So these tests are about the verifier, not the report: each one corrupts a
claim and requires a failure. A verifier that cannot fail is worse than none,
because it manufactures confidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "ml/tune/V3_2_REPORT.md"
VERIFY = ROOT / "ml/tune/verify_v3_2_report.py"

#: (label, pattern, replacement) -- each must make verification fail.
CORRUPTIONS = [
    ("eval_v3 metric", r"\| `json_parse_rate` \| 0\.9932 \| 0\.9795 \| 0\.9589 \| \*\*0\.9178\*\*",
     "| `json_parse_rate` | 0.9932 | 0.9795 | 0.9589 | **0.9999**"),
    ("frozen metric", r"\| `object_exact` \| 0\.9422 \| 0\.9696 \| 0\.9848 \| \*\*0\.9483\*\*",
     "| `object_exact` | 0.9422 | 0.9696 | 0.9848 | **0.9999**"),
    ("category figure", r"\| E \| 1\.0000 \| 0\.1667 \| 0\.5000 \| -0\.5000 \|",
     "| E | 1.0000 | 0.1667 | 0.9000 | -0.1000 |"),
    ("delta column", r"\| `hallucination_rate` \| 0\.0152 \| 0\.0456 \| 0\.0578 \| \*\*0\.0213\*\* \| -0\.0365",
     "| `hallucination_rate` | 0.0152 | 0.0456 | 0.0578 | **0.0213** | -0.9999"),
    ("adapter sha256", r"fbc8dbc089cc4d0ae91dbb93b912220e5b6bb1f25412fa4681ef31f4d155c2d5",
     "0" * 64),
    ("release decision", r"must not be shipped", "must be shipped"),
    ("json-failure row",
     r"^\| `ev3-025` \| SPLIT \| \*\*SPLIT\*\* \| JSONFAIL \| 1645 \|$", ""),
]


def _run(report: Path) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "ml.tune.verify_v3_2_report", "--report", str(report)],
        cwd=ROOT, capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


@pytest.fixture(scope="module")
def good() -> str:
    if not REPORT.exists():
        pytest.skip("V3_2_REPORT.md not present")
    return REPORT.read_text(encoding="utf-8")


def test_clean_report_passes(tmp_path, good):
    out = tmp_path / "report.md"
    out.write_text(good, encoding="utf-8")
    rc, text = _run(out)
    assert rc == 0, text
    assert "PASS" in text


def test_it_checks_a_meaningful_number_of_claims(tmp_path, good):
    """Guards the failure mode that started this module: dispatch that matches
    nothing while still reporting success."""
    out = tmp_path / "claims.json"
    p = subprocess.run(
        [sys.executable, "-m", "ml.tune.verify_v3_2_report",
         "--report", str(REPORT), "--json-out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    # every one of the three per-arm metric tables must contribute 5 checks per
    # metric (4 arms + delta); a dropped table shows up as a drop well below
    # this floor
    assert data["n_checks"] >= 190, data["n_checks"]
    assert data["failed"] == 0


@pytest.mark.parametrize("label,pattern,replacement", CORRUPTIONS,
                         ids=[c[0] for c in CORRUPTIONS])
def test_corruption_is_detected(tmp_path, good, label, pattern, replacement):
    import re
    corrupt, n = re.subn(pattern, replacement, good, flags=re.M)
    assert n >= 1, f"pattern for {label!r} no longer matches the report; the test is stale"
    out = tmp_path / "report.md"
    out.write_text(corrupt, encoding="utf-8")
    rc, text = _run(out)
    assert rc == 1, f"{label}: verifier did not fail\n{text}"
    assert "FAIL" in text


def test_section_numbers_are_not_load_bearing(tmp_path, good):
    """Renumbering the report must not disable verification.

    This is the exact regression that motivated the file: dispatch keyed on
    section numbers stopped matching when a section was inserted.
    """
    shifted = good.replace("### 7.", "### 9.")
    out = tmp_path / "report.md"
    out.write_text(shifted, encoding="utf-8")
    rc, text = _run(out)
    assert rc == 0, f"renumbering changed the verdict:\n{text}"