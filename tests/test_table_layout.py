"""Offline tests for form-table layout repair (cell shift / continuation)."""

from __future__ import annotations

import unittest

import pandas as pd

from pdf_extractor.exporters import compose_document_txt
from pdf_extractor.table_layout import (
    demote_false_data_headers,
    realign_shifted_outline_rows,
    repair_form_table,
    stitch_outline_continuation_tables,
)


class TestRealignShiftedRows(unittest.TestCase):
    def test_money_leaked_into_outline_column(self) -> None:
        df = pd.DataFrame(
            [
                ["I", "Tong so", "10.620.000", "10.370.000", "", "", "250.000"],
                ["250.000", "1.1.2", "Phat tien theo Ban an", "", "", "", ""],
            ],
            columns=["A", "B", "1", "2", "3", "4", "5"],
        )
        fixed = realign_shifted_outline_rows(df)
        self.assertEqual(fixed.iloc[1, 0], "1.1.2")
        self.assertEqual(fixed.iloc[1, 1], "Phat tien theo Ban an")
        self.assertEqual(fixed.iloc[1, 2], "250.000")

    def test_outline_shifted_right_one_column(self) -> None:
        df = pd.DataFrame(
            [
                ["", "1.1.3", "Truy thu tien", "", ""],
                ["", "1.3", "Thu khan cap", "", ""],
            ],
            columns=["A", "B", "C", "D", "E"],
        )
        fixed = realign_shifted_outline_rows(df)
        self.assertEqual(fixed.iloc[0, 0], "1.1.3")
        self.assertEqual(fixed.iloc[0, 1], "Truy thu tien")
        self.assertEqual(fixed.iloc[1, 0], "1.3")
        self.assertEqual(fixed.iloc[1, 1], "Thu khan cap")


class TestDemoteFalseHeaders(unittest.TestCase):
    def test_continuation_page_promoted_first_row(self) -> None:
        df = pd.DataFrame(
            [
                ["2", "Cac khoan theo don", "", ""],
                ["2.1", "Bang tien", "", ""],
            ],
            columns=["1.3.2", "Thu tai san khan cap", "Column_3", "Column_4"],
        )
        fixed = demote_false_data_headers(df)
        self.assertEqual(list(fixed.columns), ["Column_1", "Column_2", "Column_3", "Column_4"])
        self.assertEqual(fixed.iloc[0, 0], "1.3.2")
        self.assertEqual(fixed.iloc[0, 1], "Thu tai san khan cap")
        self.assertEqual(fixed.iloc[1, 0], "2")


class TestStitchContinuation(unittest.TestCase):
    def test_stitches_outline_continuation_across_pages(self) -> None:
        page1 = pd.DataFrame(
            [
                ["Uy thac", "Tra don", "", ""],
                ["A", "B", "1", "2"],
                ["1.3", "Thu khan cap", "", ""],
                ["1.3.1", "Thu tien", "", ""],
            ],
            columns=["", "", "so TT", "Tieu chi"],
        )
        page2 = pd.DataFrame(
            [
                ["2", "Theo don", "", ""],
                ["2.1", "Bang tien", "", ""],
            ],
            columns=["1.3.2", "Thu tai san", "Column_3", "Column_4"],
        )
        out = stitch_outline_continuation_tables([page1, page2])
        self.assertEqual(len(out), 1)
        codes = [out[0].iloc[r, 0] for r in range(len(out[0]))]
        self.assertIn("1.3.1", codes)
        self.assertIn("1.3.2", codes)
        self.assertIn("2", codes)

    def test_does_not_stitch_unrelated_tables(self) -> None:
        a = pd.DataFrame([["1", "alpha"]], columns=["Code", "Name"])
        b = pd.DataFrame([["Hue", "54"]], columns=["City", "Zip"])
        out = stitch_outline_continuation_tables([a, b])
        self.assertEqual(len(out), 2)


class TestRepairPipelineInExporter(unittest.TestCase):
    def test_compose_realigns_and_stitches_without_marker_tables(self) -> None:
        md = "Mau so: B06\n\nDon vi: Phu Tho\n"
        t1 = pd.DataFrame(
            [
                ["Uy thac THA", "Tra don THA", "", "", ""],
                ["A", "B", "1", "2", "3"],
                ["1.1", "Cac khoan", "10.620.000", "10.370.000", "250.000"],
                ["250.000", "1.1.2", "Phat tien", "", ""],
                ["", "1.1.3", "Truy thu", "", ""],
            ],
            columns=["", "", "so TT", "Tieu chi", "Tong"],
        )
        t2 = pd.DataFrame(
            [["2", "Theo don", "", "", ""]],
            columns=["1.3.2", "Thu tai san", "Column_3", "Column_4", "Column_5"],
        )
        txt = compose_document_txt(md, [t1, t2], apply_ocr_cleanup=False)
        # One continuous table preferred over two.
        self.assertEqual(txt.count("1.1.2"), 1)
        self.assertIn("| 1.1.2", txt)
        self.assertIn("| 1.1.3", txt)
        self.assertIn("| 1.3.2", txt)
        self.assertIn("| 2 ", txt)
        # Money must not remain in the outline column.
        self.assertNotIn("| 250.000     | 1.1.2", txt)
        self.assertNotIn("| 250.000 | 1.1.2", txt)


class TestAbsorbHeaderRows(unittest.TestCase):
    def test_absorbs_label_and_code_rows_into_empty_slots(self) -> None:
        df = pd.DataFrame(
            [
                ["Uy thac", "Tra don", "Dinh chi", "Mien giam", "", "", "", ""],
                ["A", "B", "1", "2", "3", "4", "5", "6"],
                ["I", "Tong phai thu", "100", "", "", "", "", ""],
            ],
            columns=["", "", "", "so TT", "Tieu chi", "Tong", "CHV", ""],
        )
        fixed = repair_form_table(df)
        self.assertEqual(fixed.iloc[0, 0], "I")
        cols = [str(c) for c in fixed.columns]
        # Primary header slid left; labels packed into remaining empties.
        self.assertTrue(cols[0].startswith("so TT") or "so TT" in cols[0])
        self.assertTrue(any("Uy thac" in c for c in cols))
        # Labels must not be smashed onto the STT column alone.
        self.assertNotIn("so TT Uy thac", cols[0])


if __name__ == "__main__":
    unittest.main()
