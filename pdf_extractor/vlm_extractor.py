"""
State-of-the-Art table extraction using a local Vision-Language Model (VLM).

This module REPLACES the coordinate-based pipeline (`coordinate_extractor.py`,
removed) and supersedes the PP-Structure pipeline (`paddle_extractor.py`) as
the primary table-heavy-document route. Instead of hardcoded coordinates,
anchor text, or even a dedicated table-structure-recognition model, this
module hands the raw page IMAGE directly to a general-purpose VLM
(`Qwen/Qwen2-VL-7B-Instruct` by default) and asks it to return the table as
strict JSON.

Design goals:
    - Enterprise-grade, object-oriented: a single `VLMTableExtractor` class
      owns model/processor lifecycle, prompting, inference, and parsing.
    - GPU-first: assumes a CUDA-equipped server. Weights are loaded in
      `bfloat16` (falls back to `float16`) to minimize VRAM usage.
    - Fail-safe JSON parsing: VLMs occasionally "hallucinate" formatting
      (markdown fences, trailing commentary, truncated JSON). Every parsing
      step is wrapped so ONE bad page never crashes the whole document run.

Pipeline:
    1. `VLMConfig`                      -> all tunable knobs in one place
       (model name, device, dtype, generation params).
    2. `VLMTableExtractor._load_model()` -> lazy-load the model + processor
       once, reused across pages/documents.
    3. `render_pdf_to_images()`          -> PDF -> list[PIL.Image] (high-res).
    4. `_build_prompt()`                 -> strict "return ONLY JSON" prompt.
    5. `_run_inference()`                -> chat-template + generate().
    6. `_parse_json_response()`          -> robust `json.loads` with fallback
       stripping of markdown fences / preamble.
    7. `_to_dataframe()`                 -> JSON array of row-objects -> DataFrame.
    8. `clean_ocr_errors()`              -> reuse the existing OCR fix-up dict.
    9. `export_to_excel()`               -> write to `output_tables.xlsx`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from PIL import Image

from .exporters import export_tables_preview
from .ocr_cleanup import clean_ocr_errors

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "output_tables.xlsx"

# The strict extraction prompt. Kept literal/verbatim per spec -- any VLM
# tends to drift into markdown/prose without this level of explicitness.
TABLE_EXTRACTION_PROMPT = (
    "Extract the table from this image. Return ONLY a valid JSON array of "
    "objects representing the table rows. Do not include any markdown "
    "formatting, preamble, or explanation."
)


class VLMExtractionError(Exception):
    """Raised when the VLM pipeline cannot be initialized or run."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class VLMConfig:
    """
    All tunable knobs for the VLM pipeline in one place.

    Generation parameters (how to adjust them):
        - `temperature`: 0.0-2.0. Lower (e.g. 0.0-0.2) is STRONGLY recommended
          for structured-data extraction like this -- it makes the model's
          output deterministic and reduces hallucinated rows/columns. Raise
          it only if you deliberately want more "creative" (i.e. less
          faithful) reconstructions, which is almost never desirable here.
        - `max_new_tokens`: caps how many tokens the model may generate.
          Increase this if your tables are large (many rows/columns) and you
          see truncated/incomplete JSON in the logs; decrease it to speed up
          inference on small tables. As a rule of thumb, budget ~15-25 tokens
          per table cell.
        - `do_sample`: set to False (paired with `temperature` effectively
          ignored) for fully greedy/deterministic decoding -- the safest
          default for data-extraction tasks. Set True + a low temperature if
          you want a small amount of controlled randomness.
    """

    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda"  # Assumes a GPU-equipped server, per deployment target.
    torch_dtype_name: str = "bfloat16"  # "bfloat16" preferred; falls back to "float16".

    # --- Generation parameters (see class docstring above for tuning advice) ---
    max_new_tokens: int = 2048
    temperature: float = 0.1
    do_sample: bool = False

    # --- Page rendering ---
    render_dpi: int = 300  # High-res rendering improves the VLM's ability to read small text.

    # --- Prompt ---
    prompt: str = field(default=TABLE_EXTRACTION_PROMPT)


# ---------------------------------------------------------------------------
# PDF -> Image rendering
# ---------------------------------------------------------------------------
def render_pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 300,
    page_indices: Optional[List[int]] = None,
) -> List[Image.Image]:
    """
    Rasterize specific pages (or all pages) of a PDF into high-resolution PIL Images.

    Uses PyMuPDF (`fitz`) rather than `pdf2image` so no external Poppler
    binary needs to be installed on the GPU server -- swap this out for
    `pdf2image.convert_from_path()` if your deployment already standardizes
    on Poppler.

    Args:
        pdf_path: Path to the input PDF.
        dpi: Rendering resolution. VLMs benefit from higher resolution than
            classic OCR (300 DPI is a good default for dense tables).
        page_indices: Optional list of 0-based page indices to render. If
            `None` (default), every page is rendered. Restricting this to a
            small subset (e.g. only pages a cheap pdfplumber scan flagged as
            containing a table) is what lets the orchestrator avoid wasting
            GPU time on pages that are pure prose.

    Returns:
        One PIL.Image (RGB) per requested page, in page order.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    try:
        indices = page_indices if page_indices is not None else range(doc.page_count)
        for i in indices:
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(image)
    finally:
        doc.close()

    logger.info("Rendered %d page(s) from %s at %d DPI (page_indices=%s).", len(images), pdf_path, dpi, page_indices)
    return images


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------
class VLMTableExtractor:
    """
    Object-oriented wrapper around a local Vision-Language Model used purely
    for table-to-JSON extraction.

    The heavy model/processor are loaded LAZILY on first use and cached on
    the instance, so constructing a `VLMTableExtractor()` is cheap -- only
    calling `.extract(...)` (or `.load()` explicitly) touches the GPU/VRAM.
    """

    def __init__(self, config: Optional[VLMConfig] = None) -> None:
        self.config = config or VLMConfig()
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """
        Load the VLM + its processor onto the GPU, in bfloat16/float16.

        Raises:
            VLMExtractionError: If `torch`/`transformers` are missing, or if
                no CUDA device is available.
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise VLMExtractionError(
                "transformers/torch (and Qwen2-VL support) are not installed. "
                "Run: pip install torch transformers accelerate qwen-vl-utils"
            ) from exc

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise VLMExtractionError(
                "VLMConfig.device='cuda' but no CUDA device is available on this machine."
            )

        # bfloat16 is preferred on modern (Ampere+) GPUs for better numerical
        # stability than float16, at the same VRAM cost. Fall back gracefully
        # if the requested dtype string is invalid.
        dtype = getattr(torch, self.config.torch_dtype_name, torch.bfloat16)

        # Clear leftover CUDA fragments (e.g. after Marker unload) before the
        # large contiguous allocation that from_pretrained needs.
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

        free_gib = None
        if self.config.device == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / (1024**3)
            logger.info(
                "GPU VRAM before VLM load: %.2f GiB free / %.2f GiB total",
                free_gib,
                total_bytes / (1024**3),
            )

        logger.info(
            "Loading VLM '%s' on %s (dtype=%s)...",
            self.config.model_name,
            self.config.device,
            dtype,
        )
        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map=self.config.device,
            )
        except torch.OutOfMemoryError as exc:
            hint = (
                f" Only ~{free_gib:.1f} GiB free before load."
                if free_gib is not None
                else ""
            )
            raise VLMExtractionError(
                "CUDA OOM while loading the VLM."
                + hint
                + " Kill other GPU processes (`nvidia-smi`, then `kill -9 <PID>`), "
                "or use a smaller model e.g. --model Qwen/Qwen2-VL-2B-Instruct. "
                f"Original error: {exc}"
            ) from exc
        self._processor = AutoProcessor.from_pretrained(self.config.model_name)
        logger.info("VLM ready.")

    # ------------------------------------------------------------------
    # Prompting + inference
    # ------------------------------------------------------------------
    def _build_messages(self, image: Image.Image) -> List[dict]:
        """Build a Qwen2-VL-style chat message list embedding the page image + prompt."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.config.prompt},
                ],
            }
        ]

    def _run_inference(self, image: Image.Image) -> str:
        """
        Run one forward pass: page image + strict prompt -> raw text response.

        Generation parameters (`max_new_tokens`, `temperature`, `do_sample`)
        are read straight from `self.config` -- tune them there rather than
        editing this method (see `VLMConfig` docstring for guidance).
        """
        import torch

        self.load()

        messages = self._build_messages(image)
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
            )

        # Strip the input prompt tokens off the front of the generated sequence
        # so we only decode the model's NEW output.
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self._processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    # ------------------------------------------------------------------
    # Robust JSON parsing (VLMs can hallucinate formatting)
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Remove ```json ... ``` / ``` ... ``` fences some VLMs add despite
        being told not to. Returns the content between the first and last
        fence if fences are present, otherwise the original text.
        """
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        return fence_match.group(1) if fence_match else text

    @staticmethod
    def _extract_json_array_substring(text: str) -> str:
        """
        Best-effort recovery of just the `[...]` JSON array from a response
        that may still contain leading/trailing prose despite the prompt.
        """
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def _parse_json_response(self, raw_text: str) -> Optional[List[dict]]:
        """
        Parse the VLM's raw text output into a list of row-dicts.

        Wrapped in try/except around `json.loads` per spec -- if the model
        hallucinates invalid JSON, this logs a warning and returns `None`
        instead of raising, so the caller can skip this page gracefully.
        """
        candidate = self._strip_markdown_fences(raw_text)
        candidate = self._extract_json_array_substring(candidate)

        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("VLM response was not valid JSON (%s). Raw response: %.200s", exc, raw_text)
            return None

        if not isinstance(data, list):
            logger.warning("VLM JSON was valid but not a list of row objects -- skipping.")
            return None

        return data

    @staticmethod
    def _to_dataframe(rows: List[dict]) -> Optional[pd.DataFrame]:
        """Convert a JSON array of row-objects into a DataFrame; None if empty."""
        if not rows:
            return None
        try:
            return pd.DataFrame(rows)
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to build a DataFrame from parsed JSON rows: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Per-page + full-document extraction
    # ------------------------------------------------------------------
    def extract_table_from_image(self, image: Image.Image) -> Optional[pd.DataFrame]:
        """
        Run the full single-page pipeline: inference -> JSON parse -> DataFrame.

        Returns `None` (never raises) if the page had no table, or if the
        VLM's output could not be parsed -- callers should treat `None` as
        "skip this page" rather than a fatal error.
        """
        try:
            raw_text = self._run_inference(image)
        except Exception as exc:  # pragma: no cover - defensive: GPU/runtime errors
            logger.warning("VLM inference failed on a page: %s", exc)
            return None

        rows = self._parse_json_response(raw_text)
        if rows is None:
            return None

        return self._to_dataframe(rows)

    def extract_pages(
        self,
        pdf_path: str | Path,
        page_indices: Optional[List[int]] = None,
        apply_ocr_cleanup: bool = True,
    ) -> List[pd.DataFrame]:
        """
        Extract tables from a specific subset of pages (0-based indices), in memory.

        This is the method the content-driven orchestrator
        (`unified_pipeline.py`) calls: it first uses a cheap `pdfplumber`
        scan to find which pages physically contain a table, then passes
        ONLY those page indices here -- the VLM never runs on pages that are
        pure prose. Pass `page_indices=None` to process every page (used by
        the standalone `extract()` / CLI entrypoint below).

        Does NOT write any file -- callers own export (so the orchestrator
        can combine results from multiple backends into one workbook).

        Returns:
            One cleaned DataFrame per requested page that yielded a usable
            table (may be an empty list -- not an error condition on its own).
        """
        pdf_path = Path(pdf_path)

        try:
            page_images = render_pdf_to_images(pdf_path, dpi=self.config.render_dpi, page_indices=page_indices)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise VLMExtractionError(f"Failed to render '{pdf_path}' to images: {exc}") from exc

        dataframes: List[pd.DataFrame] = []
        for image in page_images:
            df = self.extract_table_from_image(image)
            if df is None or df.empty:
                continue

            if apply_ocr_cleanup:
                df = clean_ocr_errors(df)

            dataframes.append(df)

        return dataframes

    def extract(
        self,
        pdf_path: str | Path,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        apply_ocr_cleanup: bool = True,
    ) -> List[pd.DataFrame]:
        """
        Full standalone pipeline: PDF -> every page -> VLM JSON extraction ->
        DataFrames -> OCR cleanup -> Excel export.

        For content-driven, page-restricted extraction (the orchestrator's
        use case), call `extract_pages()` directly instead.

        Args:
            pdf_path: Path to the input PDF.
            output_path: Destination .xlsx path.
            apply_ocr_cleanup: If True (default), run every DataFrame through
                `clean_ocr_errors()` before exporting.

        Returns:
            One cleaned DataFrame per page that yielded a usable table (may
            be an empty list -- not an error condition on its own).
        """
        dataframes = self.extract_pages(pdf_path, page_indices=None, apply_ocr_cleanup=apply_ocr_cleanup)
        self.export_to_excel(dataframes, output_path)
        return dataframes

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    @staticmethod
    def export_to_excel(dataframes: List[pd.DataFrame], output_path: str | Path) -> Path:
        """
        Write Excel + CSV + Markdown preview (VS Code-friendly companions).

        Prefer opening `*_preview.md` or `*_table_*.csv` in VS Code;
        open `.xlsx` with LibreOffice / Excel.
        """
        written = export_tables_preview(dataframes, output_path)
        return written["xlsx"]


# ---------------------------------------------------------------------------
# Module-level convenience function (what `router.py` calls)
# ---------------------------------------------------------------------------
# A single shared extractor instance so the (large) VLM is loaded once per
# process and reused across every `extract()` call, instead of once per PDF.
_default_extractor: Optional[VLMTableExtractor] = None


def _get_default_extractor() -> VLMTableExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = VLMTableExtractor()
    return _default_extractor


def extract(
    pdf_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    apply_ocr_cleanup: bool = True,
    config: Optional[VLMConfig] = None,
) -> List[pd.DataFrame]:
    """
    Module-level entrypoint mirroring `VLMTableExtractor.extract()`.

    Uses a process-wide cached `VLMTableExtractor` (so the model is only
    loaded once) unless a custom `config` is supplied, in which case a
    dedicated, non-cached extractor is created for that call.
    """
    extractor = VLMTableExtractor(config) if config is not None else _get_default_extractor()
    return extractor.extract(pdf_path, output_path=output_path, apply_ocr_cleanup=apply_ocr_cleanup)


# ---------------------------------------------------------------------------
# CLI (standalone usage / debugging, independent of the router)
# ---------------------------------------------------------------------------
def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Table extraction via a local Vision-Language Model (Qwen2-VL by "
            "default), running on CUDA. No coordinates, no anchors, no "
            "per-document configuration required."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Path to the output .xlsx file.")
    parser.add_argument("--model", default=VLMConfig.model_name, help="HuggingFace model id.")
    parser.add_argument("--max-new-tokens", type=int, default=VLMConfig.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=VLMConfig.temperature)
    parser.add_argument("--no-ocr-cleanup", action="store_true", help="Skip the OCR post-processing pass.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = VLMConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    try:
        dataframes = VLMTableExtractor(config).extract(
            args.input,
            output_path=args.output,
            apply_ocr_cleanup=not args.no_ocr_cleanup,
        )
    except (FileNotFoundError, VLMExtractionError) as exc:
        logging.error(str(exc))
        return 1

    print(f"Done. Exported {len(dataframes)} table(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
