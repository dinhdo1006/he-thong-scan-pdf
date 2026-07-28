"""Tests for B06 hierarchical Excel export (merged parent/child headers)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from pdf_extractor.b06_excel import drop_semantic_helpers, is_b06_excel_candidate, write_b06_workbook
from pdf_extractor.table_layout import _B06_CANONICAL_HEADERS


class TestB06Excel(unittest.TestCase):
    def _sample_df(self) -> pd.DataFrame:
        rows = [
            [1, "I", "số phải thu", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000", True, "N/A", ""],
            [1, "1", "Các khoản chủ động", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000", True, "OK", ""],
            [2, "1.1", "  Thu nộp", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000", True, "OK", ""],
            [3, "1.1.1", "    Án phí", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000", True, "N/A", ""],
        ]
        cols = ["Cấp"] + list(_B06_CANONICAL_HEADERS) + ["STT_valid", "Sum_check", "STT_gap"]
        return pd.DataFrame(rows, columns=cols)

    def test_detects_b06_candidate(self) -> None:
        self.assertTrue(is_b06_excel_candidate(self._sample_df()))

    def test_drop_helpers_removes_cap_and_validators(self) -> None:
        out = drop_semantic_helpers(self._sample_df())
        names = [str(c).lower() for c in out.columns]
        self.assertNotIn("cấp", names)
        self.assertNotIn("stt_valid", names)
        self.assertNotIn("sum_check", names)
        self.assertNotIn("stt_gap", names)
        self.assertEqual(out.iloc[0, 1], "Số phải thu")

    def test_writes_merged_group_header_without_helpers(self) -> None:
        df = self._sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.xlsx"
            write_b06_workbook([df], path)
            wb = load_workbook(path)
            ws = wb.active
            # No Cap column — Số TT is column A
            self.assertEqual(ws.cell(1, 1).value, "Số TT")
            self.assertIn("Trong đó Chấp hành viên", str(ws.cell(1, 4).value))
            self.assertEqual(ws.cell(2, 4).value, "Ủy thác THA")
            self.assertEqual(ws.cell(3, 1).value, "A")
            self.assertEqual(ws.cell(3, 4).value, "2")
            self.assertEqual(ws.cell(3, 9).value, "7")
            ranges = {str(r) for r in ws.merged_cells.ranges}
            self.assertTrue(any("D1:G1" in r for r in ranges))
            # No helper headers beyond col 9
            self.assertIsNone(ws.cell(1, 10).value)
            self.assertEqual(ws.cell(4, 1).value, "I")
            self.assertTrue(ws.cell(4, 1).font.bold)
            self.assertEqual(ws.cell(4, 2).value, "Số phải thu")
            self.assertEqual(ws.cell(7, 1).value, "1.1.1")


if __name__ == "__main__":
    unittest.main()
