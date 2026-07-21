"""Offline tests for export_document (.docx) -- no GPU, no PDF needed."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pdf_extractor.exporters import export_document


class TestExportDocument(unittest.TestCase):
    def test_writes_docx_with_text_and_table(self) -> None:
        from docx import Document

        df = pd.DataFrame(
            [["10.620.000", "10.370.000", "250.000"]],
            columns=["c1", "c2", "c3"],
        )
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "output.docx"
            export_document("Some prose text.\n\nSecond paragraph.", [df], out_path)

            self.assertTrue(out_path.is_file())

            doc = Document(str(out_path))
            self.assertEqual(len(doc.tables), 1)
            table = doc.tables[0]
            # 1 header row + 1 data row, 3 columns -- no merged/dropped cells.
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(len(table.columns), 3)
            self.assertEqual(table.rows[0].cells[0].text, "c1")
            self.assertEqual(table.rows[1].cells[0].text, "10.620.000")
            self.assertEqual(table.rows[1].cells[1].text, "10.370.000")
            self.assertEqual(table.rows[1].cells[2].text, "250.000")

            full_text = "\n".join(p.text for p in doc.paragraphs)
            self.assertIn("Some prose text.", full_text)
            self.assertIn("Second paragraph.", full_text)

    def test_writes_placeholder_when_no_tables(self) -> None:
        from docx import Document

        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "output.docx"
            export_document("Just text, no tables.", [], out_path)

            doc = Document(str(out_path))
            self.assertEqual(len(doc.tables), 0)
            full_text = "\n".join(p.text for p in doc.paragraphs)
            self.assertIn("Không phát hiện bảng", full_text)


if __name__ == "__main__":
    unittest.main()
