"""Offline tests for unified plain-text export -- no GPU, no PDF needed."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pdf_extractor.exporters import (
    compose_document_txt,
    dataframe_to_plaintext_table,
    resolve_output_path,
    save_unified_txt,
)


SAMPLE_MD = """# Report

Intro paragraph.

| Name | Age |
|------|-----|
| Alice | 30 |

More text after table.
"""


class TestComposeDocumentTxt(unittest.TestCase):
    def test_uses_extracted_table_when_available(self) -> None:
        extracted = [pd.DataFrame([["99", "88"]], columns=["Name", "Age"])]
        txt = compose_document_txt(SAMPLE_MD, extracted, apply_ocr_cleanup=False)
        self.assertIn("Intro paragraph", txt)
        self.assertIn("More text after table", txt)
        self.assertIn("99\t88", txt)
        self.assertNotIn("Alice", txt)

    def test_falls_back_to_marker_table_when_extraction_empty(self) -> None:
        txt = compose_document_txt(SAMPLE_MD, [], apply_ocr_cleanup=False)
        self.assertIn("Alice", txt)
        self.assertIn("30", txt)
        self.assertNotIn("|---|", txt)

    def test_preserves_reading_order(self) -> None:
        txt = compose_document_txt(SAMPLE_MD, [], apply_ocr_cleanup=False)
        intro_pos = txt.find("Intro paragraph")
        table_pos = txt.find("Alice")
        after_pos = txt.find("More text after")
        self.assertLess(intro_pos, table_pos)
        self.assertLess(table_pos, after_pos)


class TestResolveOutputPath(unittest.TestCase):
    def test_file_path(self) -> None:
        with TemporaryDirectory() as tmp:
            p = resolve_output_path(Path(tmp) / "out.txt")
            self.assertEqual(p.name, "out.txt")

    def test_directory_gets_default_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            p = resolve_output_path(Path(tmp) / "folder")
            self.assertEqual(p.name, "output.txt")
            self.assertEqual(p.parent.name, "folder")


class TestDataframeToPlaintextTable(unittest.TestCase):
    def test_omits_generic_header_row(self) -> None:
        df = pd.DataFrame([["1", "value"]], columns=["col", "col_1"])
        txt = dataframe_to_plaintext_table(df)
        self.assertEqual(txt, "1\tvalue")


class TestSaveUnifiedTxt(unittest.TestCase):
    def test_writes_utf8_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.txt"
            save_unified_txt("Xin chào\n", path)
            self.assertEqual(path.read_text(encoding="utf-8"), "Xin chào\n")


if __name__ == "__main__":
    unittest.main()
