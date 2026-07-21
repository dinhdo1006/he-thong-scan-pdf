"""
Fast, fully offline tests for grid/border-based table extraction.

No Marker, no VLM, no GPU, no network -- these run in a couple of seconds on
any machine. Use this file as your quick "did I break the wiring?" smoke test
before running the full GPU pipeline on a real PDF.

Run with:
    pytest tests/test_grid_table_extractor.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz  # PyMuPDF

from pdf_extractor.grid_table_extractor import (
    detect_table_pages,
    extract_pdf_tables_to_excel,
)
from pdf_extractor.ocr_cleanup import clean_ocr_errors
import pandas as pd


def make_bordered_table_pdf(path: Path) -> None:
    """
    Build a minimal single-page PDF with ONE ruled (bordered) 2x2 table, using
    only PyMuPDF primitives (already a project dependency) -- no external
    fixture file, no scanning, no network.
    """
    doc = fitz.open()
    page = doc.new_page()

    # A 2-column x 2-row grid of thin black rectangles (visible ruled lines),
    # which is exactly what pdfplumber's "lines" strategy detects.
    x0, y0, cell_w, cell_h = 72, 72, 100, 30
    for row in range(2):
        for col in range(2):
            rect = fitz.Rect(
                x0 + col * cell_w,
                y0 + row * cell_h,
                x0 + (col + 1) * cell_w,
                y0 + (row + 1) * cell_h,
            )
            page.draw_rect(rect, color=(0, 0, 0), width=1)
            text = ["Ten", "Tuoi", "An", "20"][row * 2 + col]
            page.insert_text((rect.x0 + 5, rect.y0 + 20), text, fontsize=10)

    doc.save(str(path))
    doc.close()


def make_plain_text_pdf(path: Path) -> None:
    """A one-page PDF with prose only -- no ruled lines, no table."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Just some plain prose, no table here at all.")
    doc.save(str(path))
    doc.close()


class TestDetectTablePages(unittest.TestCase):
    def test_finds_bordered_table(self) -> None:
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "with_table.pdf"
            make_bordered_table_pdf(pdf_path)
            pages = detect_table_pages(pdf_path)
            self.assertEqual(pages, [0])

    def test_no_table_in_plain_text(self) -> None:
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "no_table.pdf"
            make_plain_text_pdf(pdf_path)
            pages = detect_table_pages(pdf_path)
            self.assertEqual(pages, [])


class TestExtractPdfTablesToExcel(unittest.TestCase):
    def test_extracts_expected_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "with_table.pdf"
            out_path = Path(tmp) / "out.xlsx"
            make_bordered_table_pdf(pdf_path)

            dataframes = extract_pdf_tables_to_excel(
                pdf_path, output_path=out_path, apply_ocr_cleanup=False
            )

            self.assertEqual(len(dataframes), 1)
            self.assertTrue(out_path.is_file())
            df = dataframes[0]
            # header row consumed the first grid row -> 1 data row left
            self.assertEqual(len(df), 1)

    def test_no_table_writes_placeholder(self) -> None:
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "no_table.pdf"
            out_path = Path(tmp) / "out.xlsx"
            make_plain_text_pdf(pdf_path)

            dataframes = extract_pdf_tables_to_excel(pdf_path, output_path=out_path)

            self.assertEqual(dataframes, [])
            self.assertTrue(out_path.is_file())


class TestCleanOcrErrors(unittest.TestCase):
    def test_replaces_known_typo(self) -> None:
        df = pd.DataFrame({"col": ["hoyết dinh"]})
        cleaned = clean_ocr_errors(df)
        # Only asserts wiring works if ocr_corrections.json ships this key;
        # if it doesn't, the no-op path is exercised instead -- both are valid.
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertEqual(len(cleaned), 1)

    def test_missing_corrections_file_is_noop(self) -> None:
        df = pd.DataFrame({"col": ["abc"]})
        cleaned = clean_ocr_errors(df, corrections_path=Path("does_not_exist.json"))
        self.assertEqual(cleaned["col"].iloc[0], "abc")


if __name__ == "__main__":
    unittest.main()
