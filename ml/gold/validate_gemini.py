"""Local validation of Gemini-generated annotation outputs.

Validates future Gemini batch outputs WITHOUT calling Gemini or any API: the
validator checks structure, IDs, text integrity and JSON Schema conformance,
then optionally merges the validated annotations into a single unreviewed
proposals file.

Supported inputs:

* ``--input <file>`` with ``--batch <batch_id>``  -- one batch output (Format
  A: ``{"batch_id": ..., "annotations": [...]}``, or Format B: a plain JSON
  array of annotation objects).
* ``--input-dir <dir>`` -- all batch outputs together; every batch must be
  known, IDs unique across batches, none missing, no extras, total == 400.

Exit code is non-zero when any validation error is found.

Run::

    python3 -m ml.gold.validate_gemini --input <file> --batch batch-001
    python3 -m ml.gold.validate_gemini --input-dir <dir> --merge-output ml/data/gold/gemini_proposals.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

MERGE_STATUS = "gemini_proposed_unreviewed"

ALLOWED_TOP_FIELDS = ("batch_id", "annotations")
ALLOWED_ITEM_FIELDS = ("id", "annotation", "text")

FORMAT_A = "format_a"
FORMAT_B = "format_b"


class ValidationError:
    def __init__(self, category: str, record_id: str | None, path: str | None, message: str):
        self.category = category
        self.record_id = record_id
        self.path = path
        self.message = message

    def to_dict(self):
        return {
            "category": self.category,
            "record_id": self.record_id,
            "path": self.path,
            "message": self.message,
        }

    def __str__(self):
        bits = [f"category={self.category}"]
        if self.record_id:
            bits.append(f"id={self.record_id}")
        if self.path:
            bits.append(f"path={self.path}")
        bits.append(self.message)
        return " ".join(bits)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(data_dir: Path) -> dict:
    path = data_dir / "gold" / "gemini_batches" / "manifest.json"
    return load_json(path)


def load_batch_expected(data_dir: Path, batch_id: str) -> dict:
    path = data_dir / "gold" / "gemini_batches" / f"{batch_id}.json"
    return load_json(path)


def load_schema(data_dir: Path) -> dict:
    path = data_dir / "gold" / "annotation_schema.json"
    return load_json(path)


def make_validator(schema: dict) -> Draft7Validator:
    return Draft7Validator(schema)


def _error_path(error: jsonschema.ValidationError) -> str:
    return error.json_path or "/".join(str(p) for p in error.absolute_path) or "$"


def detect_format(payload: Any, errors: list) -> str | None:
    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            return FORMAT_A
        errors.append(ValidationError("format", None, None,
                                      "object payload missing 'annotations' list"))
        return None
    if isinstance(payload, list):
        return FORMAT_B
    errors.append(ValidationError("format", None, None,
                                  "payload must be an object or an array"))
    return None


def validate_payload(
    payload: Any,
    fmt: str,
    expected_ids: set[str],
    expected_texts: dict[str, str],
    data_dir: Path,
    expected_batch_id: str | None,
    validator: Draft7Validator | None = None,
) -> tuple[bool, list, dict[str, dict]]:
    """Validate one parsed Gemini output payload.

    Returns ``(ok, errors, id_to_annotation)``.
    """
    errors: list = []
    annotations: dict[str, dict] = {}

    if fmt == FORMAT_A:
        top_keys = set(payload.keys())
        if expected_batch_id is not None:
            given = payload.get("batch_id")
            if given != expected_batch_id:
                errors.append(ValidationError(
                    "wrong_batch", None, None,
                    f"payload batch_id={given!r}, expected {expected_batch_id!r}"))
        if not isinstance(payload.get("batch_id"), str):
            errors.append(ValidationError("wrong_batch", None, None,
                                          "missing/type-mismatched top-level batch_id"))
        for key in top_keys - set(ALLOWED_TOP_FIELDS):
            errors.append(ValidationError("unexpected_field", None, key,
                                          f"unexpected top-level field {key!r}"))
        items = payload.get("annotations", [])
    else:
        items = payload

    seen: set[str] = set()
    for i, item in enumerate(items):
        path_base = f"annotations[{i}]" if fmt == FORMAT_A else f"[{i}]"
        if not isinstance(item, dict):
            errors.append(ValidationError("format", None, path_base,
                                          "annotation item is not an object"))
            continue
        for key in set(item.keys()) - set(ALLOWED_ITEM_FIELDS):
            errors.append(ValidationError("unexpected_field", None,
                                          f"{path_base}.{key}",
                                          f"unexpected field {key!r}"))
        record_id = item.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(ValidationError("missing_id", None, path_base,
                                          "annotation item missing string 'id'"))
            continue
        if record_id not in expected_ids:
            errors.append(ValidationError("unknown_id", record_id, path_base,
                                          f"id {record_id!r} not in expected batch"))
        if record_id in seen:
            errors.append(ValidationError("duplicate_id", record_id, path_base,
                                          f"id {record_id!r} appears more than once"))
        seen.add(record_id)

        if "annotation" not in item or not isinstance(item["annotation"], dict):
            errors.append(ValidationError("missing_annotation", record_id,
                                          f"{path_base}.annotation",
                                          "missing/type-mismatched 'annotation' object"))
            continue
        annotation = item["annotation"]

        # Optional text integrity check (only when Gemini echoes text back).
        if "text" in item:
            actual = item["text"]
            if not isinstance(actual, str):
                errors.append(ValidationError("changed_text", record_id,
                                              f"{path_base}.text", "text is not a string"))
            elif record_id in expected_texts and actual != expected_texts[record_id]:
                errors.append(ValidationError("changed_text", record_id,
                                              f"{path_base}.text",
                                              "text does not match the original"))

        if validator is not None:
            for schema_error in validator.iter_errors(annotation):
                errors.append(ValidationError(
                    "schema", record_id,
                    f"{path_base}.annotation.{_error_path(schema_error)}",
                    schema_error.message,
                ))

        annotations[record_id] = annotation

    for missing in sorted(expected_ids - seen):
        errors.append(ValidationError("missing_id", missing, None,
                                      f"expected id {missing!r} is missing"))
    return not errors, errors, annotations


def _batch_from_path(path: Path) -> str:
    return path.stem  # batch-001.json -> batch-001


def validate_file(
    path: Path,
    data_dir: Path,
    expected_batch_id: str | None = None,
) -> tuple[bool, list, dict[str, dict]]:
    """Validate a single Gemini output file against its expected batch."""
    errors: list = []
    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        return False, [ValidationError("json_parse", None, None, str(exc))], {}
    except OSError as exc:
        return False, [ValidationError("json_parse", None, None, str(exc))], {}

    if expected_batch_id is None:
        expected_batch_id = _batch_from_path(path)

    manifest = load_manifest(data_dir)
    ids_per_batch = manifest["ids_per_batch"]
    if expected_batch_id not in ids_per_batch:
        return False, [
            ValidationError("unknown_batch", None, None,
                            f"batch {expected_batch_id!r} not in manifest")
        ], {}

    expected_ids = set(ids_per_batch[expected_batch_id])
    batch_data = load_batch_expected(data_dir, expected_batch_id)
    expected_texts = {r["id"]: r["text"] for r in batch_data["records"]}

    fmt = detect_format(payload, errors)
    if fmt is None:
        return False, errors, {}
    validator = make_validator(load_schema(data_dir))
    return validate_payload(payload, fmt, expected_ids, expected_texts,
                            data_dir, expected_batch_id, validator)


def validate_input_dir(
    input_dir: Path, data_dir: Path
) -> tuple[bool, list, dict[str, dict]]:
    """Validate every batch output present in ``input_dir`` together."""
    manifest = load_manifest(data_dir)
    batch_ids = manifest["batch_ids"]
    errors: list = []
    all_annotations: dict[str, dict] = {}
    seen_globally: dict[str, str] = {}
    expected_total = manifest["total_records"]
    found = 0

    for batch_id in batch_ids:
        path = input_dir / f"{batch_id}.json"
        if not path.exists():
            errors.append(ValidationError("missing_batch", None, None,
                                          f"no output file for {batch_id}"))
            continue
        ok, file_errors, annotations = validate_file(path, data_dir, batch_id)
        errors.extend(file_errors)
        if not ok:
            continue
        for record_id in annotations:
            if record_id in seen_globally:
                errors.append(ValidationError(
                    "cross_batch_duplicate", record_id, None,
                    f"id {record_id!r} already seen in {seen_globally[record_id]}"))
            else:
                seen_globally[record_id] = batch_id
        all_annotations.update(annotations)
        found += len(annotations)

    if found != expected_total:
        errors.append(ValidationError("incomplete", None, None,
                                      f"found {found} annotations, expected {expected_total}"))
    return not errors, errors, all_annotations


def load_annotation_set(data_dir: Path) -> list[dict]:
    path = data_dir / "gold" / "annotation_set.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def merge_proposals(
    data_dir: Path, annotations: dict[str, dict], output: Path
) -> int:
    """Merge validated annotations into an unreviewed proposals file.

    Preserves the original annotation-set record order and metadata, and adds
    ``status = "gemini_proposed_unreviewed"``. Never touches
    ``annotation_set.jsonl``.
    """
    records = load_annotation_set(data_dir)
    merged = []
    missing = [r["id"] for r in records if r["id"] not in annotations]
    if missing:
        raise RuntimeError(
            f"cannot merge: no annotation for ids: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    order = {r["id"]: i for i, r in enumerate(records)}
    extra_ids = set(annotations) - set(order)
    if extra_ids:
        raise RuntimeError(f"cannot merge: unexpected ids: {sorted(extra_ids)[:10]}")

    for rec in records:
        merged.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "source_kind": rec["source_kind"],
                "source_split": rec["source_split"],
                "selection_method": rec["selection_method"],
                "is_multitopic_candidate": rec["is_multitopic_candidate"],
                "annotation": annotations[rec["id"]],
                "status": MERGE_STATUS,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        for rec in merged:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python3 -m ml.gold.validate_gemini",
        description="Validate Gemini-generated annotation outputs locally.",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--batch", type=str, default=None,
                        help="expected batch id, e.g. batch-001")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--merge-output", type=Path, default=None)
    parser.add_argument("--data-dir", default="ml/data", type=Path)
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("provide --input <file> or --input-dir <dir>")
    if args.input is not None and args.input_dir is not None:
        parser.error("provide only one of --input / --input-dir")

    errors: list = []
    annotations: dict[str, dict] = {}

    if args.input is not None:
        ok, errors, annotations = validate_file(args.input, args.data_dir, args.batch)
    else:
        ok, errors, annotations = validate_input_dir(args.input_dir, args.data_dir)

    for e in errors:
        print(str(e))
    if errors:
        print(f"Validation FAILED: {len(errors)} error(s)")
        sys.exit(1)

    print(f"Validation OK: {len(annotations)} annotation(s)")

    if args.merge_output is not None:
        count = merge_proposals(args.data_dir, annotations, args.merge_output)
        print(f"Merged {count} proposals -> {args.merge_output}")


if __name__ == "__main__":
    main()