"""Offline tests for PaddleOCR HTML table helpers -- no model download needed."""

from __future__ import annotations

import unittest

from pdf_extractor.paddle_extractor import (
    _iter_table_htmls_from_page,
    _pred_html_from_table_res,
    html_to_dataframe,
)


class HtmlHelpersTests(unittest.TestCase):
    def test_html_to_dataframe_keeps_dotted_thousands(self) -> None:
        html = """
        <table>
          <tr><th>TT</th><th>Số tiền</th></tr>
          <tr><td>1</td><td>10.620.000</td></tr>
          <tr><td>1.1</td><td>250.000</td></tr>
        </table>
        """
        df = html_to_dataframe(html)
        self.assertIsNotNone(df)
        assert df is not None
        self.assertEqual(list(df.iloc[0]), ["1", "10.620.000"])
        self.assertEqual(list(df.iloc[1]), ["1.1", "250.000"])

    def test_pred_html_from_dict_shapes(self) -> None:
        self.assertEqual(
            _pred_html_from_table_res({"pred_html": "<table></table>"}),
            "<table></table>",
        )
        self.assertEqual(
            _pred_html_from_table_res({"html": {"pred": "<table>a</table>"}}),
            "<table>a</table>",
        )
        self.assertEqual(
            _pred_html_from_table_res({"res": {"html": "<table>b</table>"}}),
            "<table>b</table>",
        )

    def test_iter_table_htmls_from_page_dict(self) -> None:
        page = {
            "table_res_list": [
                {"pred_html": "<table><tr><td>1</td></tr></table>"},
                {"html": {"pred": "<table><tr><td>2</td></tr></table>"}},
            ]
        }
        htmls = _iter_table_htmls_from_page(page)
        self.assertEqual(len(htmls), 2)
        self.assertIn("<td>1</td>", htmls[0])
        self.assertIn("<td>2</td>", htmls[1])


if __name__ == "__main__":
    unittest.main()
