"""Lightweight unit tests for Markdown table parsing (no Marker models needed)."""

from __future__ import annotations

import unittest

from pdf_extractor.markdown_parser import MarkdownBlockParser, SEPARATOR_RE


SAMPLE_MD = """# Report

Intro paragraph with a pipe | that is not a table.

| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |

More text after the first table.

| A | B | C |
| ---: | :---: | --- |
| 1 | 2 | 3 |

Trailing notes.
"""


class TestSeparatorRegex(unittest.TestCase):
    def test_standard_separator(self) -> None:
        self.assertTrue(SEPARATOR_RE.match("|---|---|"))
        self.assertTrue(SEPARATOR_RE.match("| --- | :---: | ---: |"))
        self.assertTrue(SEPARATOR_RE.match("---|---"))

    def test_non_separator(self) -> None:
        self.assertFalse(SEPARATOR_RE.match("| Name | Age |"))
        self.assertFalse(SEPARATOR_RE.match("just text"))
        self.assertFalse(SEPARATOR_RE.match("| - |"))  # fewer than 3 dashes


class TestMarkdownBlockParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = MarkdownBlockParser()

    def test_extracts_two_tables(self) -> None:
        result = self.parser.parse(SAMPLE_MD)
        self.assertEqual(len(result.tables), 2)

        t1 = result.tables[0]
        self.assertEqual(list(t1.columns), ["Name", "Age"])
        self.assertEqual(len(t1), 2)
        self.assertEqual(t1.iloc[0]["Name"], "Alice")

        t2 = result.tables[1]
        self.assertEqual(list(t2.columns), ["A", "B", "C"])
        self.assertEqual(t2.iloc[0]["A"], "1")

    def test_text_excludes_tables_keeps_prose(self) -> None:
        result = self.parser.parse(SAMPLE_MD)
        self.assertIn("Intro paragraph", result.text)
        self.assertIn("More text after", result.text)
        self.assertIn("Trailing notes", result.text)
        self.assertNotIn("|---|", result.text)
        self.assertNotIn("| Alice |", result.text)

    def test_lone_pipe_not_treated_as_table(self) -> None:
        result = self.parser.parse("Price | notes without separator\n\nHello\n")
        self.assertEqual(len(result.tables), 0)
        self.assertIn("Price | notes", result.text)


if __name__ == "__main__":
    unittest.main()
