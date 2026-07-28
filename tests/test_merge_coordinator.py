"""Unit tests for merge coordinator."""

from __future__ import annotations

import time
import unittest

from pdf_extractor.merge_coordinator import MergeCoordinator


class TestMergeCoordinator(unittest.TestCase):
    def test_waits_for_tables(self) -> None:
        c = MergeCoordinator(require_tables=True, timeout_s=999)
        self.assertIsNone(
            c.on_page(
                document_id="doc1",
                json_path="preprocessed/doc1/page_1.json",
                page_number=1,
                total_pages=2,
            )
        )
        self.assertIsNone(
            c.on_page(
                document_id="doc1",
                json_path="preprocessed/doc1/page_2.json",
                page_number=2,
                total_pages=2,
            )
        )
        ready = c.on_tables(
            document_id="doc1",
            tables_json_path="tables/doc1/structured_tables.json",
            status="success",
        )
        self.assertIsNotNone(ready)
        payload = c.merge_payload(ready)
        self.assertEqual(payload["merged_pages_count"], 2)
        self.assertTrue(payload["tables_json_path"].endswith("structured_tables.json"))

    def test_timeout_allows_merge_without_tables(self) -> None:
        c = MergeCoordinator(require_tables=True, timeout_s=0.01)
        c.on_page(
            document_id="doc2",
            json_path="preprocessed/doc2/page_1.json",
            page_number=1,
            total_pages=1,
        )
        time.sleep(0.02)
        ready = c.on_page(
            document_id="doc2",
            json_path="preprocessed/doc2/page_1.json",
            page_number=1,
            total_pages=1,
        )
        self.assertIsNotNone(ready)
        self.assertEqual(ready.table_status, "timeout")


if __name__ == "__main__":
    unittest.main()
