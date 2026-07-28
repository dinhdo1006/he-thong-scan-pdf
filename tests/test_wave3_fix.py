"""Tests for PDF outline enrichment + STT column detection after Cap."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from pdf_extractor.outline_enrichment import (
    PdfOutlineRow,
    enrich_table_from_pdf_outline,
    extract_outline_rows_from_pdf,
)
from pdf_extractor.semantic_validator import annotate_dataframe_semantics, detect_stt_column
from pdf_extractor.table_layout import is_outline_code


class TestIsOutlineCode(unittest.TestCase):
    def test_rejects_money_and_dates(self) -> None:
        self.assertFalse(is_outline_code("10.620.000"))
        self.assertFalse(is_outline_code("17.6.2010"))
        self.assertFalse(is_outline_code("12"))
        self.assertTrue(is_outline_code("1.1.1"))
        self.assertTrue(is_outline_code("I"))


class TestDetectSttIgnoresCap(unittest.TestCase):
    def test_prefers_stt_over_cap(self) -> None:
        df = pd.DataFrame(
            {
                "Cấp": [1, 1, 2, 3],
                "Số TT A": ["I", "1", "1.1", "1.1.1"],
                "Mo_ta": ["a", "b", "c", "d"],
            }
        )
        self.assertEqual(detect_stt_column(df), "Số TT A")


class TestEnrichFromOutline(unittest.TestCase):
    def test_inserts_missing_an_phi_and_clears_orphan_money(self) -> None:
        df = pd.DataFrame(
            [
                ["1.1", "Các khoản thu, nộp Nhà nước", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000"],
                ["1.1.2", "Phạt tiền theo Bản án", "", "", "", "", "", "", "250.000"],
                ["1.1.3", "Truy thu tiền", "", "", "", "", "", "", ""],
            ],
            columns=["STT", "Mo_ta", "1", "2", "3", "4", "5", "6", "7"],
        )
        outline = [
            PdfOutlineRow(
                "1.1",
                "Các khoản thu, nộp Nhà nước",
                ("10.620.000", "10.370.000", "250.000", "250.000"),
            ),
            PdfOutlineRow(
                "1.1.1",
                "Án phí",
                ("10.620.000", "10.370.000", "250.000", "250.000"),
            ),
            PdfOutlineRow("1.1.2", "Phạt tiền theo Bản án", ()),
            PdfOutlineRow("1.1.3", "Truy thu tiền", ()),
        ]
        out = enrich_table_from_pdf_outline(df, outline)
        stts = [str(x) for x in out["STT"].tolist()]
        self.assertIn("1.1.1", stts)
        self.assertLess(stts.index("1.1.1"), stts.index("1.1.2"))
        row111 = out[out["STT"] == "1.1.1"].iloc[0]
        self.assertEqual(row111["Mo_ta"], "Án phí")
        self.assertEqual(row111["1"], "10.620.000")
        self.assertEqual(row111["6"], "250.000")
        row112 = out[out["STT"] == "1.1.2"].iloc[0]
        self.assertEqual(str(row112["7"]).strip(), "")

    def test_cd02_pdf_extracts_key_rows(self) -> None:
        pdf = Path(r"d:\pdf test\CD_02_2011_021.pdf")
        if not pdf.is_file():
            self.skipTest("CD_02 sample PDF not present")
        rows = extract_outline_rows_from_pdf(pdf)
        by = {r.stt: r for r in rows}
        self.assertIn("1.1.1", by)
        self.assertIn("Án phí", by["1.1.1"].label)
        self.assertGreaterEqual(len(by["1.1.1"].amounts), 4)
        self.assertIn("1.2.2", by)
        self.assertIn("1.1.6", by)


class TestAnnotateUsesRealStt(unittest.TestCase):
    def test_no_false_stt_on_cap_values(self) -> None:
        df = pd.DataFrame(
            [
                ["I", "Tong", "10.620.000"],
                ["1", "Chu dong", "10.620.000"],
                ["1.1", "Thu nop", "10.620.000"],
                ["1.1.1", "An phi", "10.620.000"],
            ],
            columns=["STT", "Mo_ta", "Tien"],
        )
        out = annotate_dataframe_semantics(df)
        self.assertTrue(bool(out["STT_valid"].all()))


class TestB06RepairPipeline(unittest.TestCase):
    def test_b06_header_and_hierarchy_shape(self) -> None:
        df = pd.DataFrame(
            [
                ["I", "Số phải thu thi hành án", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000"],
                ["1", "Các khoản chủ động T.H.A", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000"],
                ["1.1", "Các khoản thu, nộp Nhà nước", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000"],
                ["1.1.1", "Án phí", "10.620.000", "10.370.000", "", "", "", "250.000", "250.000"],
            ],
            columns=[
                "Ủy thác THA A",
                "Trả đơn THA B",
                "Đình chỉ THA 1",
                "Số TT Miễn, giảm THA 2",
                "Tiêu chí trong quyết định thi hành án 3",
                "Tổng số tiền, giá trị tài sản phải thi hành 4",
                "Trong đó Chấp hành viên 5",
                "6",
                "7",
            ],
        )
        from pdf_extractor.table_layout import repair_form_table

        out = annotate_dataframe_semantics(repair_form_table(df))
        self.assertEqual(out.columns[0], "Số TT A")
        self.assertEqual(out.columns[1], "Tiêu chí trong quyết định thi hành án B")
        self.assertEqual(out.iloc[3]["Số TT A"], "1.1.1")
        self.assertEqual(
            str(out.iloc[3]["Tiêu chí trong quyết định thi hành án B"]).strip(),
            "Án phí",
        )
        self.assertNotIn("Cấp", out.columns)


if __name__ == "__main__":
    unittest.main()
