"""Tests for Wave 1 page-region assemble + Wave 2 VL markdown parsing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pdf_extractor.exporters import (
    assemble_document_by_page_regions,
    compose_document_txt,
    merge_table_preserving_form,
)
from pdf_extractor.paddle_vl_extractor import markdown_tables_to_dataframes
from pdf_extractor.text_fallback import PageProseRegion


class TestPageRegionAssemble(unittest.TestCase):
    def test_signatures_after_table_not_before(self) -> None:
        regions = [
            PageProseRegion(
                page_index=0,
                above="Mau so: B06\nDon vi: Phu Tho",
                below="Nguyen Van A\nPHO CUC TRUONG",
                table_bbox=(50.0, 100.0, 500.0, 700.0),
            )
        ]
        table = pd.DataFrame([["I", "So phai thu", "1000"]], columns=["STT", "Mo_ta", "Tien"])
        table.attrs["page_no"] = 0

        txt = compose_document_txt(
            "",
            [table],
            apply_ocr_cleanup=False,
            page_regions=regions,
        )
        header_pos = txt.find("Mau so")
        table_pos = txt.find("So phai thu")
        sig_pos = txt.find("Nguyen Van A")
        self.assertLess(header_pos, table_pos)
        self.assertLess(table_pos, sig_pos)

    def test_prefer_extracted_skips_garbled_marker_merge(self) -> None:
        marker = pd.DataFrame([["a", "b"]], columns=["hoyết Tình", "Erở nh"])
        extracted = pd.DataFrame([["1", "Thu an phi"]], columns=["So", "Noi dung"])
        merged = merge_table_preserving_form(marker, extracted, prefer_extracted=True)
        self.assertEqual(list(merged.columns), ["So", "Noi dung"])
        self.assertEqual(merged.iloc[0, 1], "Thu an phi")

    def test_garbled_marker_without_flag_also_uses_extracted(self) -> None:
        marker = pd.DataFrame([["a", "b"]], columns=["hoyết Tình", "Erở nh"])
        extracted = pd.DataFrame([["1", "Thu an phi"]], columns=["So", "Noi dung"])
        merged = merge_table_preserving_form(marker, extracted)
        self.assertEqual(list(merged.columns), ["So", "Noi dung"])

    def test_assemble_by_regions_orders_multi_page(self) -> None:
        regions = [
            PageProseRegion(0, above="Header p1", below="", table_bbox=(0, 0, 10, 10)),
            PageProseRegion(1, above="", below="Footer p2", table_bbox=(0, 0, 10, 10)),
        ]
        t0 = pd.DataFrame([["1"]], columns=["STT"])
        t0.attrs["page_no"] = 0
        t1 = pd.DataFrame([["2"]], columns=["STT"])
        t1.attrs["page_no"] = 1
        blocks = assemble_document_by_page_regions(regions, [t0, t1], apply_ocr_cleanup=False)
        kinds = [b.kind for b in blocks]
        self.assertEqual(kinds.count("table"), 2)
        texts = [str(b.content) for b in blocks if b.kind == "text"]
        self.assertTrue(any("Header p1" in t for t in texts))
        self.assertTrue(any("Footer p2" in t for t in texts))


class TestPaddleVLMarkdown(unittest.TestCase):
    def test_markdown_tables_to_dataframes(self) -> None:
        md = """# Page

| STT | Mo_ta | Tien |
|-----|-------|------|
| I | So phai thu | 10.620.000 |
| 1.1 | Cac khoan | 250.000 |
"""
        dfs = markdown_tables_to_dataframes(md)
        self.assertEqual(len(dfs), 1)
        self.assertEqual(dfs[0].shape[0], 2)
        self.assertIn("10.620.000", str(dfs[0].iloc[0, 2]))

    def test_paddle_vl_available_safe(self) -> None:
        from pdf_extractor.paddle_vl_extractor import paddle_vl_available

        # Must not raise whether or not PaddleOCRVL is installed.
        self.assertIsInstance(paddle_vl_available(), bool)


class TestUnifiedVLGate(unittest.TestCase):
    def test_skip_paddle_vl_disables_tier1(self) -> None:
        from pdf_extractor.unified_pipeline import UnifiedPDFPipeline

        pipe = UnifiedPDFPipeline.__new__(UnifiedPDFPipeline)
        pipe.skip_paddle_vl = True
        pipe.force_paddle_vl = False
        self.assertFalse(pipe._should_try_paddle_vl())

    def test_force_paddle_vl_enables_tier1(self) -> None:
        from pdf_extractor.unified_pipeline import UnifiedPDFPipeline

        pipe = UnifiedPDFPipeline.__new__(UnifiedPDFPipeline)
        pipe.skip_paddle_vl = True
        pipe.force_paddle_vl = True
        with patch("pdf_extractor.unified_pipeline._PADDLE_VL_IMPORTABLE", True), patch(
            "pdf_extractor.unified_pipeline.paddle_vl_extract_with_pages", lambda *a, **k: []
        ), patch("pdf_extractor.unified_pipeline.paddle_vl_available", return_value=True):
            self.assertTrue(pipe._should_try_paddle_vl())


if __name__ == "__main__":
    unittest.main()
