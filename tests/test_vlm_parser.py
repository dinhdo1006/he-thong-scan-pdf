"""Offline unit tests for generic VLM JSON -> DataFrame conversion (no GPU)."""

from __future__ import annotations

import unittest

import pandas as pd

from pdf_extractor.vlm_extractor import VLMTableExtractor


class TestPayloadToDataframes(unittest.TestCase):
    def test_headers_and_rows_object(self) -> None:
        payload = {
            "headers": ["A", "B", "C"],
            "rows": [["1", "2", "3"], ["4", "5", "6"]],
        }
        dfs = VLMTableExtractor._payload_to_dataframes(payload)
        self.assertEqual(len(dfs), 1)
        self.assertEqual(list(dfs[0].columns), ["A", "B", "C"])
        self.assertEqual(dfs[0].iloc[0]["A"], "1")
        self.assertEqual(dfs[0].iloc[1]["C"], "6")

    def test_list_of_table_objects(self) -> None:
        payload = [
            {"headers": ["X", "Y"], "rows": [["a", "b"]]},
            {"headers": ["P"], "rows": [["q"]]},
        ]
        dfs = VLMTableExtractor._payload_to_dataframes(payload)
        self.assertEqual(len(dfs), 2)

    def test_does_not_merge_adjacent_numeric_cells(self) -> None:
        """Regression: two number cells must stay two columns, never one string."""
        payload = {
            "headers": ["col1", "col2", "col3"],
            "rows": [["10.620.000", "10.370.000", "250.000"]],
        }
        df = VLMTableExtractor._payload_to_dataframes(payload)[0]
        self.assertEqual(df.iloc[0]["col1"], "10.620.000")
        self.assertEqual(df.iloc[0]["col2"], "10.370.000")
        self.assertEqual(df.iloc[0]["col3"], "250.000")

    def test_pure_2d_grid(self) -> None:
        payload = [["H1", "H2"], ["v1", "v2"]]
        dfs = VLMTableExtractor._payload_to_dataframes(payload)
        self.assertEqual(list(dfs[0].columns), ["H1", "H2"])
        self.assertEqual(dfs[0].iloc[0]["H1"], "v1")

    def test_row_objects_still_supported(self) -> None:
        payload = [{"name": "An", "age": "20"}, {"name": "Binh", "age": "21"}]
        dfs = VLMTableExtractor._payload_to_dataframes(payload)
        self.assertEqual(len(dfs), 1)
        self.assertEqual(dfs[0].iloc[0]["name"], "An")

    def test_tables_wrapper(self) -> None:
        payload = {"tables": [{"headers": ["A"], "rows": [["1"]]}]}
        dfs = VLMTableExtractor._payload_to_dataframes(payload)
        self.assertEqual(len(dfs), 1)


class TestStitchContinuation(unittest.TestCase):
    def test_same_width_is_stitched(self) -> None:
        a = pd.DataFrame([["1", "x"]], columns=["c1", "c2"])
        b = pd.DataFrame([["2", "y"]], columns=["Column_1", "Column_2"])
        out = VLMTableExtractor._stitch_continuation_tables([a, b])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2)
        self.assertEqual(list(out[0].columns), ["c1", "c2"])

    def test_different_width_stays_separate(self) -> None:
        a = pd.DataFrame([["1"]], columns=["c1"])
        b = pd.DataFrame([["2", "3"]], columns=["c1", "c2"])
        out = VLMTableExtractor._stitch_continuation_tables([a, b])
        self.assertEqual(len(out), 2)


class TestParseJsonResponse(unittest.TestCase):
    def test_strips_fences(self) -> None:
        raw = '```json\n[{"headers":["A"],"rows":[["1"]]}]\n```'
        data = VLMTableExtractor()._parse_json_response(raw)
        dfs = VLMTableExtractor._payload_to_dataframes(data)
        self.assertEqual(dfs[0].iloc[0]["A"], "1")


if __name__ == "__main__":
    unittest.main()
