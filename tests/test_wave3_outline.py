"""Wave 3: outline Cap/indent, missing STT gaps, money-in-description realign."""

from __future__ import annotations

import unittest

import pandas as pd

from pdf_extractor.semantic_validator import (
    annotate_dataframe_semantics,
    insert_missing_outline_placeholders,
    outline_level,
)
from pdf_extractor.table_layout import (
    normalize_outline_token,
    realign_shifted_outline_rows,
    repair_form_table,
)


class TestNormalizeOutline(unittest.TestCase):
    def test_fraction_slash(self) -> None:
        self.assertEqual(normalize_outline_token("1⁄2"), "1.2")
        self.assertEqual(normalize_outline_token("1/2"), "1.2")


class TestMoneyInDescriptionRealign(unittest.TestCase):
    def test_money_between_stt_and_label(self) -> None:
        df = pd.DataFrame(
            [["1.1.2", "250.000", "Phat tien theo Ban an", "", ""]],
            columns=["STT", "Mo_ta", "c1", "c2", "c3"],
        )
        fixed = realign_shifted_outline_rows(df)
        self.assertEqual(fixed.iloc[0, 0], "1.1.2")
        self.assertEqual(fixed.iloc[0, 1], "Phat tien theo Ban an")
        self.assertEqual(fixed.iloc[0, 2], "250.000")


class TestMissingGapsAndCap(unittest.TestCase):
    def test_outline_level(self) -> None:
        self.assertEqual(outline_level("I"), 1)
        self.assertEqual(outline_level("1"), 1)
        self.assertEqual(outline_level("1.1"), 2)
        self.assertEqual(outline_level("1.1.2"), 3)

    def test_insert_gap_1_1_1_and_1_1_6(self) -> None:
        df = pd.DataFrame(
            [
                ["1.1", "Thu nop"],
                ["1.1.2", "Phat tien"],
                ["1.1.5", "Truy thu"],
                ["1.1.7", "Thu hoi"],
            ],
            columns=["STT", "Mo_ta"],
        )
        filled = insert_missing_outline_placeholders(df)
        stts = [str(x) for x in filled["STT"].tolist()]
        self.assertIn("1.1.1", stts)
        self.assertIn("1.1.6", stts)
        self.assertLess(stts.index("1.1.1"), stts.index("1.1.2"))
        self.assertLess(stts.index("1.1.6"), stts.index("1.1.7"))

    def test_annotate_adds_gap_without_cap(self) -> None:
        df = pd.DataFrame(
            [
                ["I", "Tong so", "10.620.000"],
                ["1", "Chu dong", "10.620.000"],
                ["1.1", "Thu nop", "10.620.000"],
                ["1.1.2", "Phat tien", "250.000"],
            ],
            columns=["STT", "Mo_ta", "Tien"],
        )
        out = annotate_dataframe_semantics(df)
        self.assertNotIn("Cấp", out.columns)
        self.assertIn("STT_gap", out.columns)
        # Placeholder 1.1.1 inserted with MISSING flag.
        gap_rows = out[out["STT"] == "1.1.1"]
        self.assertEqual(len(gap_rows), 1)
        self.assertEqual(gap_rows.iloc[0]["STT_gap"], "MISSING")
        # Description is not space-indented anymore.
        row_112 = out[out["STT"] == "1.1.2"].iloc[0]
        self.assertFalse(str(row_112["Mo_ta"]).startswith("  "))


class TestDropEmptyRows(unittest.TestCase):
    def test_repair_drops_blank_rows(self) -> None:
        df = pd.DataFrame(
            [
                ["1.1.4", "Tich thu", ""],
                ["", "", ""],
                ["1.1.5", "Truy thu", ""],
            ],
            columns=["STT", "Mo_ta", "Tien"],
        )
        fixed = repair_form_table(df)
        self.assertEqual(len(fixed), 2)


if __name__ == "__main__":
    unittest.main()
