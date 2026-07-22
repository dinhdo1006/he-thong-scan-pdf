"""Tests for PyMuPDF text fallback helpers."""

from __future__ import annotations

import unittest

from pdf_extractor.text_fallback import _header_lines_only, plaintext_as_markdown


class TextFallbackTests(unittest.TestCase):
    def test_header_lines_stop_at_money(self) -> None:
        raw = "Đơn vị: A\nBÁO CÁO\n\nI Số phải thu 10.620.000\n1 Các khoản"
        out = _header_lines_only(raw)
        self.assertIn("Đơn vị", out)
        self.assertIn("BÁO CÁO", out)
        self.assertNotIn("10.620.000", out)

    def test_plaintext_as_markdown_passthrough(self) -> None:
        self.assertEqual(plaintext_as_markdown("  hello  "), "hello")


if __name__ == "__main__":
    unittest.main()
