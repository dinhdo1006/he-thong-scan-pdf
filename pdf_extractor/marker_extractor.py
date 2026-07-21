"""Local PDF → Markdown extraction via marker-pdf (no LLM / no API)."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkerExtractionError(Exception):
    """Raised when Marker fails to convert a PDF to Markdown."""


def _free_cuda() -> None:
    """Best-effort: release unused CUDA cache so the next model can allocate."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - optional dependency / no GPU
        pass
    gc.collect()


class MarkerExtractor:
    """
    Thin wrapper around marker-pdf's PdfConverter.

    Runs 100% locally: models are loaded from disk (downloaded once on first use).
    LLM mode is explicitly disabled so no external API calls occur.
    """

    def __init__(self) -> None:
        self._converter = None

    def _ensure_converter(self) -> None:
        """Lazy-load heavy Marker models only when first needed."""
        if self._converter is not None:
            return

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except ImportError as exc:
            raise MarkerExtractionError(
                "marker-pdf is not installed. Run: pip install marker-pdf"
            ) from exc

        logger.info("Loading Marker models (local only, use_llm=False)...")
        # config forces offline local inference — never enable use_llm here
        self._converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config={"use_llm": False},
        )
        logger.info("Marker models ready.")

    def unload(self) -> None:
        """
        Drop Marker models from memory / VRAM.

        Call this after text extraction finishes and BEFORE loading the VLM
        (Qwen2-VL-7B alone needs ~14 GiB in bf16). Keeping both loaded on a
        16 GiB card almost always OOMs.
        """
        if self._converter is None:
            return
        logger.info("Unloading Marker models to free VRAM for table extraction...")
        self._converter = None
        _free_cuda()
        logger.info("Marker models unloaded.")

    def extract_markdown(self, pdf_path: str | Path) -> str:
        """
        Convert a PDF file to a Markdown string.

        Args:
            pdf_path: Path to a local PDF (scanned or native).

        Returns:
            Full document as Markdown text.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            MarkerExtractionError: If conversion fails.
        """
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise MarkerExtractionError(f"Expected a .pdf file, got: {path.suffix}")

        self._ensure_converter()

        try:
            from marker.output import text_from_rendered

            logger.info("Converting PDF with Marker: %s", path)
            rendered = self._converter(str(path))
            # Prefer the rendered.markdown property when available
            if hasattr(rendered, "markdown") and rendered.markdown:
                markdown = rendered.markdown
            else:
                text, _, _ = text_from_rendered(rendered)
                markdown = text or ""
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise MarkerExtractionError(
                f"Marker failed to convert '{path}': {exc}"
            ) from exc

        if not markdown.strip():
            raise MarkerExtractionError(
                f"Marker returned empty Markdown for '{path}'."
            )

        logger.info("Markdown extracted (%d characters).", len(markdown))
        return markdown
