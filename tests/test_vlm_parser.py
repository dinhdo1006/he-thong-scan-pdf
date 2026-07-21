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


class TestHtmlTableParsing(unittest.TestCase):
    """The primary VLM response format is now HTML -- these never touch the GPU."""

    def test_basic_table(self) -> None:
        html = (
            "<table><tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        dfs = VLMTableExtractor._extract_html_tables(html)
        self.assertEqual(len(dfs), 1)
        df = VLMTableExtractor._html_table_to_dataframe(dfs[0])
        self.assertEqual(list(df.columns), ["A", "B"])
        self.assertEqual(df.iloc[0]["A"], "1")
        self.assertEqual(df.iloc[0]["B"], "2")

    def test_does_not_merge_adjacent_numeric_cells(self) -> None:
        html = (
            "<table><tr><th>c1</th><th>c2</th><th>c3</th></tr>"
            "<tr><td>10.620.000</td><td>10.370.000</td><td>250.000</td></tr></table>"
        )
        df = VLMTableExtractor._html_table_to_dataframe(html)
        self.assertEqual(df.iloc[0]["c1"], "10.620.000")
        self.assertEqual(df.iloc[0]["c2"], "10.370.000")
        self.assertEqual(df.iloc[0]["c3"], "250.000")

    def test_empty_cells_preserved(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td></td><td>x</td></tr></table>"
        df = VLMTableExtractor._html_table_to_dataframe(html)
        self.assertEqual(df.iloc[0]["A"], "")
        self.assertEqual(df.iloc[0]["B"], "x")

    def test_colspan_header_is_expanded_not_dropped(self) -> None:
        html = (
            "<table><tr><th>A</th><th colspan='2'>B</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
        )
        df = VLMTableExtractor._html_table_to_dataframe(html)
        self.assertEqual(len(df.columns), 3)

    def test_multiple_tables_in_one_response(self) -> None:
        raw = (
            "<table><tr><td>1</td></tr></table>"
            "<table><tr><td>2</td></tr></table>"
        )
        fragments = VLMTableExtractor._extract_html_tables(raw)
        self.assertEqual(len(fragments), 2)

    def test_extract_tables_prefers_html_over_json(self) -> None:
        raw = (
            "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
            '\n[{"headers":["ignored"],"rows":[["x"]]}]'
        )
        extractor = VLMTableExtractor()
        html_dfs = extractor._parse_html_response(raw)
        self.assertEqual(len(html_dfs), 1)
        self.assertEqual(html_dfs[0].iloc[0]["A"], "1")

    def test_no_html_falls_back_to_json_shape(self) -> None:
        raw = '[{"headers":["A"],"rows":[["1"]]}]'
        extractor = VLMTableExtractor()
        self.assertEqual(extractor._parse_html_response(raw), [])
        payload = extractor._parse_json_response(raw)
        dfs = VLMTableExtractor._payload_to_dataframes(payload)
        self.assertEqual(dfs[0].iloc[0]["A"], "1")


if __name__ == "__main__":
    unittest.main()
