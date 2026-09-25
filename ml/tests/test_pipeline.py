import json
import os
import tempfile
import unittest
import csv

from ml.pipeline import run_pipeline


def write_csv(path, rows):
    fieldnames = [
        "uid", "receivedDateTime", "type", "kind", "content",
        "accrualMethod", "CATUTTC", "addressPostCode", "addressAdminUnitL1",
        "addressAdminUnitL2", "addressAdminUnitL3", "addressAdminUnitL4",
        "addressPostName", "addressThoroughfare", "addressLocatorDesignator",
        "addressLocatorBuilding", "status", "organizationName",
        "organizationId", "result",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {k: "" for k in fieldnames}
            row.update(r)
            writer.writerow(row)


def rec(uid, dt, kind, content):
    return {"uid": uid, "receivedDateTime": dt, "kind": kind, "content": content}


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmp.name, "input.csv")
        self.out_dir = os.path.join(self.tmp.name, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def _make_dataset(self):
        rows = []
        # noise rows (dropped by noise filter) — 2
        rows.append(rec("N1", "2025-01-01T09:00", "Інше",
                        "Надано номер телефону ГЛ ПФ (0800)."))
        rows.append(rec("N2", "2025-01-02T09:00", "Теплопостачання",
                        "Надано номери телефонів оператора."))
        # broad-kind rows (dropped by kind filter) — 3
        rows.append(rec("B1", "2025-01-01T09:00", "Інше", "Питання без конкретики."))
        rows.append(rec("B2", "2025-01-02T09:00", "Інші питання", "Ще щось."))
        rows.append(rec("B3", "2025-01-03T09:00", "null", "null"))
        # valid records — label "heat": 3 train, 2 val, 1 test
        for i in range(3):
            rows.append(rec(f"H{i}", f"2025-0{i + 1}-01T09:00", "Теплопостачання",
                            f"Скарга тепло t{i}"))
        rows.append(rec("HV1", "2026-02-01T09:00", "Теплопостачання", "Скарга тепло v1"))
        rows.append(rec("HV2", "2026-03-01T09:00", "Теплопостачання", "Скарга тепло v2"))
        rows.append(rec("HT1", "2026-08-01T09:00", "Теплопостачання", "Скарга тепло s1"))
        # valid records — label "water": 2 train
        for i in range(2):
            rows.append(rec(f"W{i}", f"2025-10-0{i + 1}T09:00", "Водопостачання",
                            f"Скарга вода w{i}"))
        # exact duplicate of (content, kind) of H0 (dropped by dedup) — 1
        rows.append(rec("DUP", "2025-01-01T09:00", "Теплопостачання", "Скарга тепло t0"))
        return rows

    def test_produces_expected_outputs(self):
        write_csv(self.input_path, self._make_dataset())
        report = run_pipeline(
            input_csv=self.input_path,
            out_dir=self.out_dir,
            train_start="2026-01-01",
            val_start="2026-07-01",
            min_examples=2,
        )

        for name in ("train", "validation", "test"):
            path = os.path.join(self.out_dir, f"{name}.jsonl")
            self.assertTrue(os.path.exists(path), name)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "labels.json")))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "statistics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "statistics.md")))

        self.assertEqual(report["leakage"]["total"], 0)
        self.assertEqual(report["counts"]["raw"], 14)
        self.assertEqual(report["counts"]["after_noise_filter"], 12)
        self.assertEqual(report["counts"]["after_excluding_broad_kinds"], 9)
        self.assertEqual(report["counts"]["after_deduplication"], 8)
        self.assertEqual(report["counts"]["after_min_examples"], 8)
        self.assertEqual(report["counts"]["final_splits"]["train"], 5)
        self.assertEqual(report["counts"]["final_splits"]["validation"], 2)
        self.assertEqual(report["counts"]["final_splits"]["test"], 1)

    def test_unparseable_date_fails_fast(self):
        rows = self._make_dataset()
        rows.append(rec("BAD", "2025/01/01", "Теплопостачання", "Погана дата."))
        write_csv(self.input_path, rows)
        with self.assertRaises(ValueError):
            run_pipeline(
                input_csv=self.input_path,
                out_dir=self.out_dir,
                train_start="2026-01-01",
                val_start="2026-07-01",
                min_examples=2,
            )

    def test_labels_json_sorted_and_machine_readable(self):
        write_csv(self.input_path, self._make_dataset())
        run_pipeline(
            input_csv=self.input_path,
            out_dir=self.out_dir,
            train_start="2026-01-01",
            val_start="2026-07-01",
            min_examples=2,
        )
        labels = json.load(open(os.path.join(self.out_dir, "labels.json"),
                                encoding="utf-8"))
        self.assertEqual(labels, {"Водопостачання": 0, "Теплопостачання": 1})


if __name__ == "__main__":
    unittest.main()