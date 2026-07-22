"""Tests for Windows process-isolation helpers (no ML models)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from pdf_extractor import isolated_backends
from pdf_extractor.unified_pipeline import _needs_process_isolation, _run_paddle_fallback


class IsolationHelpersTests(unittest.TestCase):
    def test_needs_isolation_is_bool(self) -> None:
        self.assertIsInstance(_needs_process_isolation(), bool)

    def test_run_paddle_fallback_uses_isolated_on_windows(self) -> None:
        fake = [pd.DataFrame({"a": ["1"]})]
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("pdf_extractor.unified_pipeline.extract_paddle_tables_isolated", return_value=fake) as mocked:
                with patch("pdf_extractor.unified_pipeline._needs_process_isolation", return_value=True):
                    frames = _run_paddle_fallback(Path("dummy.pdf"), out_dir)
            mocked.assert_called_once()
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].iloc[0, 0], "1")


if __name__ == "__main__":
    unittest.main()
