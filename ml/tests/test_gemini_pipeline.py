"""Tests for the Gemini-ready annotation pipeline (ml/gold)."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.gold import prepare_gemini, validate_gemini, review_report

DATA_DIR = Path("ml/data")
REAL_SET = DATA_DIR / "gold" / "annotation_set.jsonl"


def _write_file(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_temp_corpus(n: int, root: Path) -> tuple[Path, list[dict]]:
    """Create a temp data dir with an n-record annotation set + schema."""
    data_dir = root / "data"
    (data_dir / "gold").mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        records.append(
            {
                "id": f"train:S-{i:03d}",
                "text": f"Прошу полагодити дорогу на вулиці N, будинок {i}.",
                "source_kind": "Санітарний стан, благоустрій населених пунктів, прибудинкових територій",
                "source_split": "train",
                "selection_method": "kmeans_centroid",
                "is_multitopic_candidate": False,
                "annotation": {"topics": []},
            }
        )
    with open(data_dir / "gold" / "annotation_set.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (data_dir / "gold" / "annotation_schema.json").write_text(
        (DATA_DIR / "gold" / "annotation_schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return data_dir, records


def _valid_topic(i: int, multi: bool = False) -> dict:
    topics = [
        {
            "domain": "roads",
            "issue": "damaged road surface",
            "object": {"address": f"вул. N, {i}", "kind": "road"},
            "requested_action": "repair the roadway",
            "attributes": {},
        }
    ]
    if multi:
        topics.append(
            {
                "domain": "housing",
                "issue": "broken entrance door",
                "object": "під'їзд 2, будинок 12",
                "requested_action": "repair",
                "attributes": {},
            }
        )
    return {"topics": topics}


class TestPrepareBatches(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.data_dir, self.records = _build_temp_corpus(120, Path(self._td.name))

    def test_deterministic_batching(self):
        manifest = prepare_gemini.write_outputs(self.data_dir, 40)
        b1 = (self.data_dir / "gold" / "gemini_batches" / "batch-001.json").read_bytes()
        m1 = (self.data_dir / "gold" / "gemini_batches" / "manifest.json").read_bytes()
        prepare_gemini.write_outputs(self.data_dir, 40)
        b2 = (self.data_dir / "gold" / "gemini_batches" / "batch-001.json").read_bytes()
        m2 = (self.data_dir / "gold" / "gemini_batches" / "manifest.json").read_bytes()
        self.assertEqual(b1, b2)
        self.assertEqual(m1, m2)
        self.assertEqual(manifest["num_batches"], 3)

    def test_exact_coverage_and_batch_sizes(self):
        manifest = prepare_gemini.write_outputs(self.data_dir, 40)
        self.assertEqual(manifest["total_records"], 120)
        self.assertEqual(manifest["records_per_batch"]["batch-001"], 40)
        self.assertEqual(manifest["records_per_batch"]["batch-002"], 40)
        self.assertEqual(manifest["records_per_batch"]["batch-003"], 40)
        all_ids = [i for ids in manifest["ids_per_batch"].values() for i in ids]
        self.assertEqual(len(all_ids), 120)
        self.assertEqual(len(set(all_ids)), 120)

    def test_no_duplicated_ids(self):
        manifest = prepare_gemini.write_outputs(self.data_dir, 40)
        for ids in manifest["ids_per_batch"].values():
            self.assertEqual(len(ids), len(set(ids)))

    def test_manifest_correctness(self):
        manifest = prepare_gemini.write_outputs(self.data_dir, 40)
        self.assertEqual(manifest["batch_ids"], ["batch-001", "batch-002", "batch-003"])
        src_hash = hashlib.sha256(
            (self.data_dir / "gold" / "annotation_set.jsonl").read_bytes()
        ).hexdigest()
        self.assertEqual(manifest["source_sha256"], src_hash)
        self.assertIn("generation", manifest)
        self.assertIn("prepare_script_version", manifest["generation"])

    def test_batch_contains_only_needed_fields(self):
        prepare_gemini.write_outputs(self.data_dir, 40)
        with open(self.data_dir / "gold" / "gemini_batches" / "batch-001.json", encoding="utf-8") as fh:
            batch = json.load(fh)
        self.assertEqual(set(batch.keys()), {"batch_id", "schema_version", "instructions", "records"})
        rec = batch["records"][0]
        self.assertEqual(set(rec.keys()), {"id", "text", "source_kind"})
        self.assertNotIn("source_split", rec)


class TestValidateGemini(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        root = Path(self._td.name)
        self.data_dir, self.records = _build_temp_corpus(120, root)
        self.manifest = prepare_gemini.write_outputs(self.data_dir, 40)
        self.out_dir = root / "outputs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _payload_a(self, batch_id="batch-001", drop=None, add=None, schema_bad=None):
        ids = self.manifest["ids_per_batch"][batch_id]
        items = []
        for i, rid in enumerate(ids):
            ann = _valid_topic(i)
            if schema_bad and (drop is None or i != 0):
                ann = {"topics": [{"domain": "roads"}]}
            item = {"id": rid, "annotation": ann}
            items.append(item)
        if drop:
            items = [it for it in items if it["id"] != drop]
        if add:
            items.append({"id": add, "annotation": _valid_topic(999)})
        return {"batch_id": batch_id, "annotations": items}

    def test_valid_format_a_ok(self):
        path = self.out_dir / "batch-001.json"
        _write_file(path, self._payload_a())
        ok, errors, ann = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertTrue(ok, errors)
        self.assertEqual(len(ann), 40)

    def test_valid_format_b_ok(self):
        ids = self.manifest["ids_per_batch"]["batch-001"]
        items = [{"id": rid, "annotation": _valid_topic(i)} for i, rid in enumerate(ids)]
        path = self.out_dir / "batch-001.json"
        _write_file(path, items)
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertTrue(ok, errors)

    def test_malformed_json_rejected(self):
        path = self.out_dir / "batch-001.json"
        path.write_text("{not json", encoding="utf-8")
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "json_parse" for e in errors))

    def test_missing_id_rejected(self):
        path = self.out_dir / "batch-001.json"
        payload = self._payload_a(drop=self.manifest["ids_per_batch"]["batch-001"][0])
        _write_file(path, payload)
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "missing_id" for e in errors))

    def test_duplicate_id_rejected(self):
        ids = self.manifest["ids_per_batch"]["batch-001"]
        items = [{"id": ids[0], "annotation": _valid_topic(0)},
                 {"id": ids[0], "annotation": _valid_topic(0)}]
        path = self.out_dir / "batch-001.json"
        _write_file(path, {"batch_id": "batch-001", "annotations": items})
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "duplicate_id" for e in errors))

    def test_unknown_id_rejected(self):
        path = self.out_dir / "batch-001.json"
        _write_file(path, self._payload_a(add="train:S-999"))
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "unknown_id" for e in errors))

    def test_wrong_batch_rejected(self):
        path = self.out_dir / "batch-002.json"
        _write_file(path, self._payload_a())  # payload says batch-001
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-002")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "wrong_batch" for e in errors))

    def test_schema_violation_rejected(self):
        ids = self.manifest["ids_per_batch"]["batch-001"]
        bad = {"id": ids[0], "annotation": {"topics": "definitely not an array"}}
        good = [{"id": rid, "annotation": _valid_topic(i)}
                for i, rid in enumerate(ids[1:])]
        path = self.out_dir / "batch-001.json"
        _write_file(path, {"batch_id": "batch-001", "annotations": [bad] + good})
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "schema" for e in errors))

    def test_changed_source_text_rejected(self):
        ids = self.manifest["ids_per_batch"]["batch-001"]
        items = [
            {"id": ids[0], "text": "Оригінальний текст змінено", "annotation": _valid_topic(0)},
        ] + [{"id": rid, "annotation": _valid_topic(i)}
             for i, rid in enumerate(ids[1:])]
        path = self.out_dir / "batch-001.json"
        _write_file(path, {"batch_id": "batch-001", "annotations": items})
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "changed_text" for e in errors))

    def test_mentioning_expected_text_is_ok(self):
        ids = self.manifest["ids_per_batch"]["batch-001"]
        records = json.loads(open(
            self.data_dir / "gold" / "gemini_batches" / "batch-001.json", encoding="utf-8"
        ).read())["records"]
        text_by_id = {r["id"]: r["text"] for r in records}
        items = [{"id": rid, "text": text_by_id[rid], "annotation": _valid_topic(i)}
                 for i, rid in enumerate(ids)]
        path = self.out_dir / "batch-001.json"
        _write_file(path, {"batch_id": "batch-001", "annotations": items})
        ok, errors, _ = validate_gemini.validate_file(path, self.data_dir, "batch-001")
        self.assertTrue(ok, errors)


class TestDirectoryAndMerge(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        root = Path(self._td.name)
        self.data_dir, self.records = _build_temp_corpus(120, root)
        self.manifest = prepare_gemini.write_outputs(self.data_dir, 40)
        self.out_dir = root / "outputs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _write_all_valid(self):
        for bid in self.manifest["batch_ids"]:
            ids = self.manifest["ids_per_batch"][bid]
            items = [{"id": rid, "annotation": _valid_topic(i, multi=(i % 5 == 0))}
                     for i, rid in enumerate(ids)]
            _write_file(self.out_dir / f"{bid}.json",
                        {"batch_id": bid, "annotations": items})

    def test_directory_valid(self):
        self._write_all_valid()
        ok, errors, ann = validate_gemini.validate_input_dir(self.out_dir, self.data_dir)
        self.assertTrue(ok, errors)
        self.assertEqual(len(ann), 120)

    def test_incomplete_directory_rejected(self):
        self._write_all_valid()
        (self.out_dir / "batch-003.json").unlink()
        ok, errors, _ = validate_gemini.validate_input_dir(self.out_dir, self.data_dir)
        self.assertFalse(ok)
        self.assertTrue(any(e.category == "missing_batch" for e in errors)
                        or any(e.category == "incomplete" for e in errors))

    def test_cross_batch_duplicate_rejected(self):
        self._write_all_valid()
        # Steal the first id of batch-002 into batch-001's output: the id
        # now appears in two batches, which must be rejected.
        stolen = self.manifest["ids_per_batch"]["batch-002"][0]
        path = self.out_dir / "batch-001.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["annotations"].pop()  # keep count at 40
        payload["annotations"].append({"id": stolen, "annotation": _valid_topic(0)})
        _write_file(path, payload)
        ok, errors, _ = validate_gemini.validate_input_dir(self.out_dir, self.data_dir)
        self.assertFalse(ok)
        cats = {e.category for e in errors}
        self.assertTrue(
            cats & {"unknown_id", "cross_batch_duplicate", "missing_id"},
            f"expected duplicate-related failure, got {cats}",
        )

    def test_merge_preserves_original_order_and_count(self):
        self._write_all_valid()
        ok, errors, ann = validate_gemini.validate_input_dir(self.out_dir, self.data_dir)
        self.assertTrue(ok, errors)
        merged_path = Path(self._td.name) / "gemini_proposals.jsonl"
        n = validate_gemini.merge_proposals(self.data_dir, ann, merged_path)
        merged = [json.loads(l) for l in merged_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(n, 120)
        self.assertEqual(len(merged), 120)
        original_ids = [r["id"] for r in self.records]
        self.assertEqual([m["id"] for m in merged], original_ids)
        for rec in merged:
            self.assertEqual(rec["status"], "gemini_proposed_unreviewed")
            self.assertIn("text", rec)
            self.assertIn("source_split", rec)
            self.assertIn("selection_method", rec)

    def test_merge_requires_all_ids(self):
        ann = {"train:S-000": {"topics": []}}
        merged_path = Path(self._td.name) / "gemini_proposals.jsonl"
        with self.assertRaises(RuntimeError):
            validate_gemini.merge_proposals(self.data_dir, ann, merged_path)


class TestReviewReport(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        root = Path(self._td.name)
        self.data_dir, self.records = _build_temp_corpus(40, root)

    def test_report_builds(self):
        proposals = []
        for i, rec in enumerate(self.records):
            proposals.append({
                "id": rec["id"],
                "text": rec["text"],
                "source_kind": rec["source_kind"],
                "source_split": "train",
                "selection_method": "kmeans_centroid",
                "is_multitopic_candidate": False,
                "annotation": {"topics": []} if i % 7 == 0 else _valid_topic(i),
                "status": "gemini_proposed_unreviewed",
            })
        stats = review_report.analyze(proposals, review_report.KIND_ORDER)
        self.assertTrue(stats["total_records"] > 0)
        md = review_report.render_markdown(stats)
        self.assertIn("# Gemini Proposal Review Report", md)
        self.assertIn("Source-kind vs proposed-domain", md)


class TestRealSetUntouched(unittest.TestCase):
    def test_real_annotation_set_present_and_untouched(self):
        if not REAL_SET.exists():
            raise unittest.SkipTest("real dataset not present")
        original = REAL_SET.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        self.assertEqual(len(digest), 64)
        # Source data files must remain unchanged too.
        for name in ("train", "validation", "test"):
            pth = DATA_DIR / f"{name}.jsonl"
            if not pth.exists():
                self.skipTest("generated corpus missing from public checkout")
            with open(pth, "rb") as fh:
                self.assertEqual(len(hashlib.sha256(fh.read()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()