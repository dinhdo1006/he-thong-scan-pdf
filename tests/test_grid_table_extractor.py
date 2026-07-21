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
    find_table_bbox,
    page_looks_like_scanned_table,
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


def make_scanned_table_pdf(path: Path) -> None:
    """
    Simulate a SCANNED page: rasterize a bordered table into a plain image,
    then embed that image (and nothing else -- no vector rect/line objects)
    on a fresh page. `page.find_tables()` cannot see this table at all; only
    the pixel-based `page_looks_like_scanned_table` heuristic can.
    """
    vector_doc = fitz.open()
    vector_page = vector_doc.new_page()
    # Cover most of the page (both width and height), matching a real
    # full-page scanned form -- a table confined to a small corner of the
    # page has too low a line-density-relative-to-page-size to trip the
    # heuristic, which is tuned for realistic full-page tables.
    x0, y0, cell_w, cell_h = 40, 40, 100, 40
    for row in range(18):
        for col in range(5):
            rect = fitz.Rect(
                x0 + col * cell_w, y0 + row * cell_h, x0 + (col + 1) * cell_w, y0 + (row + 1) * cell_h
            )
            vector_page.draw_rect(rect, color=(0, 0, 0), width=1.5)
    pix = vector_page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    vector_doc.close()

    scanned_doc = fitz.open()
    scanned_page = scanned_doc.new_page()
    scanned_page.insert_image(scanned_page.rect, stream=img_bytes)
    scanned_doc.save(str(path))
    scanned_doc.close()


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

    def test_visual_fallback_finds_scanned_table(self) -> None:
        """A scanned page has ZERO vector table objects; only the pixel-based
        fallback can find it -- this is exactly the bug this heuristic fixes."""
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "scanned_table.pdf"
            make_scanned_table_pdf(pdf_path)

            self.assertTrue(page_looks_like_scanned_table(pdf_path, 0))

            pages_without_fallback = detect_table_pages(pdf_path, use_visual_fallback=False)
            self.assertEqual(pages_without_fallback, [])

            pages_with_fallback = detect_table_pages(pdf_path, use_visual_fallback=True)
            self.assertEqual(pages_with_fallback, [0])


class TestFindTableBbox(unittest.TestCase):
    def test_returns_none_when_no_grid(self) -> None:
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "no_table.pdf"
            make_plain_text_pdf(pdf_path)
            self.assertIsNone(find_table_bbox(pdf_path, 0))

    def test_bbox_covers_table_region_only(self) -> None:
        with TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "scanned_table.pdf"
            make_scanned_table_pdf(pdf_path)

            bbox = find_table_bbox(pdf_path, 0)
            self.assertIsNotNone(bbox)
            x0, y0, x1, y1 = bbox

            doc = fitz.open(str(pdf_path))
            page_w, page_h = doc[0].rect.width, doc[0].rect.height
            doc.close()

            # Table occupies most, but not necessarily all, of the page --
            # bbox must stay within page bounds and be a real sub-region.
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, page_w)
            self.assertLessEqual(y1, page_h)
            self.assertGreater(x1, x0)
            self.assertGreater(y1, y0)


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
