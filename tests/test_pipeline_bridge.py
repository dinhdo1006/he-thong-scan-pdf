"""Unit tests for structured table JSON bridge."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pdf_extractor.pipeline_bridge import (
    dataframe_to_structured,
    payload_to_tables,
    read_structured_tables_json,
    structured_to_dataframe,
    tables_to_payload,
    write_structured_tables_json,
)


class TestPipelineBridge(unittest.TestCase):
    def test_roundtrip_preserves_header_matrix(self) -> None:
        df = pd.DataFrame(
            [["I", "Tong", "100"], ["1", "Chi tiet", "50"]],
            columns=["STT", "Tieu chi", "Trong do A"],
        )
        df.attrs["header_matrix"] = [
            ["STT", "Tieu chi", "Trong do"],
            ["", "", "A"],
            ["A", "B", "1"],
        ]
        df.attrs["page_no"] = 0

        payload = tables_to_payload([df])
        self.assertEqual(payload["version"], 1)
        self.assertEqual(len(payload["tables"]), 1)
        obj = payload["tables"][0]
        self.assertEqual(obj["page"], 0)
        self.assertEqual(len(obj["header_matrix"]), 3)

        restored = payload_to_tables(payload)[0]
        self.assertEqual(list(restored.columns), list(df.columns))
        self.assertEqual(restored.values.tolist(), df.values.tolist())
        self.assertEqual(restored.attrs["header_matrix"], df.attrs["header_matrix"])
        self.assertEqual(restored.attrs["page_no"], 0)

    def test_write_read_json_file(self) -> None:
        df = pd.DataFrame([["1", "x"]], columns=["a", "b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_structured_tables_json([df], Path(tmp) / "t.json")
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("tables", raw)
            back = read_structured_tables_json(path)
            self.assertEqual(len(back), 1)
            self.assertEqual(back[0].iloc[0, 0], "1")

    def test_empty_table(self) -> None:
        obj = dataframe_to_structured(pd.DataFrame(), table_id=3)
        self.assertEqual(obj["table_id"], 3)
        self.assertEqual(obj["data"], [])
        df = structured_to_dataframe(obj)
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
