"""Machine-verifiable release-provenance and distribution-status checks.

Validates only **deterministic repository facts**. It performs no network I/O and
fabricates nothing: external claims (source licensing, public HF reachability)
are *not* checked here, they are recorded in ``DATA_PROVENANCE.md`` and
``FINAL_RELEASE_READINESS.md`` from evidence captured during the provenance
audit, with the exact commands given there so anyone can re-derive them.

What this guards against is the failure mode that actually happened: a document
asserting "no licence is recorded" while the publisher's metadata says CC BY 4.0.
It also pins the canonical identifiers and the distribution status vocabulary.

Run::

    .venv/bin/python -m ml.tune.verify_release_provenance
    .venv/bin/python -m ml.tune.verify_release_provenance -v

Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- canonical identifiers (must stay in sync with V2_SERVING / MODEL_CARD_V2)
CANONICAL_MODEL_ID = "Fenomen-Alex/ukrainian-municipal-accountability-qwen3-8b"
BASE_MODEL = "mlx-community/Qwen3-8B-4bit"
CANONICAL_LOCAL_DIR = "ml/data/tune/adapters/qwen3-8b-lora-v2-attempt10-fused"
CANONICAL_MANIFEST_KEY = "qwen3-8b-lora-v2-fused"

SOURCE_PUBLISHER = "https://data.kr-rada.gov.ua/dataset/1770a320-880f-4cd6-9840-78b9e8d93553"
SOURCE_RESOURCE = (
    "https://data.kr-rada.gov.ua/dataset/"
    "9109c3a3-980f-45d0-a83e-b60f87d583d4/resource/"
    "ddf7ffe9-d8e0-4646-93b2-b10d0b4bfe63/download/appeals.csv"
)
#: The id this project used to record, which is wrong by one character and 404s.
SOURCE_ID_TYPO = "9109c3a3-980f-45d0-a83e-b60f87d583c4"
SOURCE_LICENSE = "CC BY 4.0"

PROVENANCE_DOC = REPO / "DATA_PROVENANCE.md"
READINESS_DOC = REPO / "FINAL_RELEASE_READINESS.md"
SHA_MANIFEST = REPO / "ml/data/tune/artifact_sha256.json"

#: The only strings permitted as a distribution status.
ALLOWED_STATUSES = (
    "READY FOR PUBLIC DISTRIBUTION",
    "READY FOR PUBLIC DISTRIBUTION — PENDING HISTORY CLEANUP",
    "READY EXCEPT FOR EXTERNAL VERIFICATION",
    "BLOCKED",
)

#: Filenames that must never be committed from the upstream source dataset.
FORBIDDEN_SOURCE_FILES = ("appeals.csv", "appeals_2026-08-01.csv")

#: Claims that were true once and are now known to be false. If any of these
#: phrases reappear as a *present-tense assertion about the source*, the repo has
#: regressed onto a disproven provenance claim.
DISPROVEN_CLAIMS = (
    r"no recorded source URL",
    r"licen[cs]e is not\s+documented",
    r"has no recorded source URL",
    r"no source licen[cs]e or redistribution",
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def run_checks() -> list[Check]:
    checks: list[Check] = []
    add = lambda n, ok, d: checks.append(Check(n, bool(ok), d))  # noqa: E731

    # 1. required documents exist and are non-trivial
    for label, path, floor in (
        ("provenance doc exists", PROVENANCE_DOC, 2000),
        ("readiness doc exists", READINESS_DOC, 2000),
    ):
        size = path.stat().st_size if path.exists() else 0
        add(label, size >= floor, f"{path.relative_to(REPO)} = {size} bytes (min {floor})")

    prov = PROVENANCE_DOC.read_text(encoding="utf-8") if PROVENANCE_DOC.exists() else ""
    ready = READINESS_DOC.read_text(encoding="utf-8") if READINESS_DOC.exists() else ""

    # 2. source URL and licence recorded in the provenance doc
    add("source dataset URL recorded", SOURCE_PUBLISHER in prov, SOURCE_PUBLISHER)
    add("source resource URL recorded", SOURCE_RESOURCE in prov, "resource download URL")
    add("source licence recorded", SOURCE_LICENSE in prov, SOURCE_LICENSE)
    add("publisher named", "Kropyvnytskyi City Council" in prov, "Executive Committee")
    add("attribution obligation stated",
        "condition that the creator is appropriately credited" in prov
        or "on condition that the creator is credited" in prov
        or "condition the creator is credited" in prov,
        "CC BY attribution condition quoted")
    add("dataset-id typo documented", SOURCE_ID_TYPO in prov,
        "the previously recorded, incorrect id is called out")

    # 3. weights vs corpus licensing kept distinct
    add("weights/corpus licensing distinguished",
        "does not" in prov and "Apache-2.0" in prov and SOURCE_LICENSE in prov,
        "Apache-2.0 (weights) is not presented as covering the corpus")

    # 4. canonical identifiers agree with the serving contract
    serving = (REPO / "ml/tune/V2_SERVING.md").read_text(encoding="utf-8")
    add("canonical model id consistent",
        CANONICAL_MODEL_ID in serving and CANONICAL_MODEL_ID in ready,
        CANONICAL_MODEL_ID)
    add("base model consistent", BASE_MODEL in serving, BASE_MODEL)
    add("canonical local dir consistent",
        CANONICAL_LOCAL_DIR in serving and CANONICAL_LOCAL_DIR in ready,
        CANONICAL_LOCAL_DIR)

    # 5. expected HF hash manifest exists and covers the canonical artifact
    manifest_ok = False
    detail = "missing"
    if SHA_MANIFEST.exists():
        data = json.loads(SHA_MANIFEST.read_text(encoding="utf-8"))
        entry = data.get(CANONICAL_MANIFEST_KEY)
        if entry:
            files = entry.get("files") or {}
            needed = {"config.json", "model.safetensors", "tokenizer.json",
                      "tokenizer_config.json", "chat_template.jinja",
                      "model.safetensors.index.json"}
            manifest_ok = needed <= set(files) and all(
                re.fullmatch(r"[0-9a-f]{64}", v.get("sha256", "")) for v in files.values()
            )
            detail = f"{len(files)} files, all sha256 well-formed"
        else:
            detail = f"key {CANONICAL_MANIFEST_KEY!r} absent"
    add("HF sha256 manifest covers canonical artifact", manifest_ok, detail)

    # 6. repository does not track the upstream source CSV or model weights
    tracked = _tracked_files()
    tracked_set = set(tracked)
    leaked = [f for f in FORBIDDEN_SOURCE_FILES
              if any(t.endswith("/" + f) or t == f for t in tracked_set)]
    add("no upstream source CSV tracked", not leaked, f"leaked={leaked or 'none'}")

    weighty = [t for t in tracked
               if t.endswith((".safetensors", ".gguf", ".bin", ".pt", ".pth"))]
    add("no model weights tracked in git", not weighty, f"found={weighty or 'none'}")

    adapter = [t for t in tracked if "/adapters/" in t]
    add("no adapter weights tracked in git", not adapter, f"found={adapter or 'none'}")

    # 7. distribution status is one of the allowed values
    m = re.search(r"^[\*>_\s]*Status:\s*`([^`]+)`", ready, re.M)
    status = m.group(1).strip() if m else None
    add("distribution status declared", status is not None, f"status={status!r}")
    add("distribution status is an allowed value",
        status in ALLOWED_STATUSES, f"allowed={ALLOWED_STATUSES}")

    # 8. the disproven claim has not crept back into any tracked markdown
    md = [t for t in tracked if t.endswith(".md")
          and not t.startswith(("ml/data/tune/adapters/", "node_modules"))]
    offenders: list[str] = []
    for rel in md:
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for pat in DISPROVEN_CLAIMS:
            if re.search(pat, text, re.I):
                # allowed only where explicitly marked as the corrected error
                idx = text.lower().find(re.sub(r"[\\]", "", pat).split("|")[0][:20].lower())
                offenders.append(f"{rel}: /{pat}/")
    add("no disproven provenance claim asserted", not offenders,
        f"offenders={offenders or 'none'}")

    return checks


def main() -> None:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    checks = run_checks()
    failed = [c for c in checks if not c.ok]
    for c in checks:
        if verbose or not c.ok:
            print(f"[{'PASS' if c.ok else 'FAIL'}] {c.name}")
            print(f"         {c.detail}")
    print(f"\nrelease provenance: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(c.name for c in failed))
        raise SystemExit(1)
    print("OK: repository provenance claims are internally consistent.")


if __name__ == "__main__":
    main()