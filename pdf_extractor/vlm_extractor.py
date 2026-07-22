"""
State-of-the-Art table extraction using a local Vision-Language Model (VLM).

This module REPLACES the coordinate-based pipeline (`coordinate_extractor.py`,
removed) and supersedes the PP-Structure pipeline (`paddle_extractor.py`) as
the primary table-heavy-document route. Instead of hardcoded coordinates,
anchor text, or even a dedicated table-structure-recognition model, this
module hands the raw page IMAGE directly to a general-purpose VLM
(`Qwen/Qwen2-VL-7B-Instruct` by default) and asks it to return the table as
an HTML `<table>` (with a JSON grid as an automatic fallback shape).

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

# Fully generic: NO template name, NO fixed column list, NO form id.
# The model must reconstruct whatever grid is visually present on THIS page.
# HTML is the PRIMARY requested format -- an actual <table>/<tr>/<td> grid
# maps much more directly onto "one ruled cell = one <td>" than a JSON object
# does, and colspan/rowspan give the model an explicit way to describe merged
# header cells instead of guessing at a flattened column list.
TABLE_EXTRACTION_PROMPT = """You are reading a PAGE IMAGE. Find EVERY table by looking ONLY at the visual grid (ruled lines / cell borders) drawn on the image. Read every character from the IMAGE itself -- do not trust any pre-existing OCR text layer, it may be wrong.

Rules (apply to ANY table, any language, any layout -- never assume a named template or fixed schema):
1. Return each physical table as one HTML <table> element, in reading order.
2. One row of the printed grid = one <tr>. One printed cell = one <td> (use <th> only for a clear header row/cell).
3. Preserve the exact column count of the printed grid in every row. Never merge two adjacent cells into one value -- two separate number cells must stay two separate <td> cells.
4. A cell with no visible text is still a cell: use an empty <td></td>, never omit it. Never drop blank columns in the middle of a form.
5. Keep the visual column positions: left-to-right order of <td> must match the printed form left-to-right.
6. Keep hierarchical / outline row codes exactly as printed in the first columns (e.g. 1, 1.1, 1.1.1, II) -- each printed line is its own <tr>, never flatten parent/child rows into one cell.
7. If a header cell visually spans multiple columns or rows, use colspan/rowspan on that <th> exactly as printed -- do not flatten it into one column.
8. Return ONLY the <table>...</table> element(s). No markdown fences, no <html>/<body> wrapper, no commentary before or after.

Example of the exact shape (structure only, not real content):
<table>
<tr><th>A</th><th colspan="2">B</th></tr>
<tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""

# Appended to the HTML prompt above so a model/checkpoint that ignores the
# HTML instruction still has a well-defined generic fallback shape to answer
# in -- the parser (`_parse_vlm_response`) tries HTML first, then this JSON
# shape, on every response.
_JSON_FALLBACK_HINT = (
    "\n\nIf you cannot produce HTML for some table, return that one as a JSON object "
    'instead, shaped as {"headers": ["...", "..."], "rows": [["...", "..."]]}.'
)

TABLE_EXTRACTION_PROMPT = TABLE_EXTRACTION_PROMPT + _JSON_FALLBACK_HINT


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

    # 7B needs ~16+ GiB free for load+image; on 16 GiB cards prefer 2B unless
    # the caller forces a larger model via CLI / env.
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    fallback_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cuda"  # Assumes a GPU-equipped server, per deployment target.
    torch_dtype_name: str = "bfloat16"  # "bfloat16" preferred; falls back to "float16".

    # --- Generation parameters (see class docstring above for tuning advice) ---
    # Large / multi-column forms need more tokens; truncated JSON is worse than slower runs.
    max_new_tokens: int = 8192
    temperature: float = 0.1
    do_sample: bool = False

    # --- Page rendering ---
    # High-res rendering improves reading of small printed cells from the IMAGE
    # (we deliberately do not use the PDF's embedded text layer).
    render_dpi: int = 300
    # Default False for scanned full-page forms: aggressive crop often clips
    # multi-header B06-style grids and the VLM returns nothing usable.
    crop_to_table: bool = False

    # --- Prompt ---
    prompt: str = field(default=TABLE_EXTRACTION_PROMPT)


# ---------------------------------------------------------------------------
# PDF -> Image rendering
# ---------------------------------------------------------------------------
def render_pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 300,
    page_indices: Optional[List[int]] = None,
    crop_to_table: bool = False,
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
        crop_to_table: If True, crop each rendered page to the densest
            ruled-grid region found by `grid_table_extractor.find_table_bbox`
            (a generic, pixel-density heuristic -- no template/column
            assumptions). Pages where no such region is detected are
            rendered in full, unchanged. This removes letterhead/signature/
            stamp noise around the table so the VLM's attention/tokens focus
            on the grid itself.

    Returns:
        One PIL.Image (RGB) per requested page, in page order.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    bbox_by_page: dict[int, Any] = {}
    if crop_to_table:
        from .grid_table_extractor import find_table_bbox

        indices_for_bbox = page_indices if page_indices is not None else None
        with fitz.open(pdf_path) as _probe:
            probe_indices = indices_for_bbox if indices_for_bbox is not None else range(_probe.page_count)
        for i in probe_indices:
            try:
                bbox = find_table_bbox(pdf_path, i)
            except Exception as exc:  # pragma: no cover - defensive: rendering failure
                logger.warning("Table bbox detection failed on page %d: %s", i + 1, exc)
                bbox = None
            if bbox is not None:
                bbox_by_page[i] = bbox

    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    try:
        indices = page_indices if page_indices is not None else range(doc.page_count)
        for i in indices:
            page = doc[i]
            clip = fitz.Rect(*bbox_by_page[i]) if i in bbox_by_page else None
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, clip=clip)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(image)
            if clip is not None:
                logger.info("Page %d: cropped to detected table region %s.", i + 1, tuple(round(v, 1) for v in clip))
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
        # After a load OOM, do not retry the same model on every page / crop pass.
        self._load_failed: bool = False
        self._load_error: Optional[str] = None

    def unload(self) -> None:
        """Drop model weights and free CUDA cache."""
        import gc

        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

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
        if self._load_failed:
            raise VLMExtractionError(
                self._load_error
                or "VLM load previously failed (CUDA OOM). Skipping further VLM attempts."
            )

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
        total_gib = None
        if self.config.device == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / (1024**3)
            total_gib = total_bytes / (1024**3)
            logger.info(
                "GPU VRAM before VLM load: %.2f GiB free / %.2f GiB total",
                free_gib,
                total_gib,
            )

        model_candidates = [self.config.model_name]
        # On ~16 GiB GPUs, 7B often OOMs during shard load; try 2B automatically.
        if (
            self.config.fallback_model_name
            and self.config.fallback_model_name != self.config.model_name
        ):
            model_candidates.append(self.config.fallback_model_name)
        elif (
            total_gib is not None
            and total_gib < 20
            and "7B" in self.config.model_name
            and self.config.fallback_model_name
        ):
            model_candidates.append(self.config.fallback_model_name)

        # De-dupe while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for name in model_candidates:
            if name not in seen:
                seen.add(name)
                ordered.append(name)

        last_exc: Optional[Exception] = None
        for model_name in ordered:
            logger.info(
                "Loading VLM '%s' on %s (dtype=%s)...",
                model_name,
                self.config.device,
                dtype,
            )
            try:
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=self.config.device,
                )
                self._processor = AutoProcessor.from_pretrained(model_name)
                self.config.model_name = model_name
                logger.info("VLM ready (%s).", model_name)
                return
            except torch.OutOfMemoryError as exc:
                last_exc = exc
                logger.warning(
                    "CUDA OOM loading '%s' -- unloading partial weights and trying next model.",
                    model_name,
                )
                self.unload()
                continue
            except Exception as exc:
                last_exc = exc
                logger.warning("Failed loading '%s': %s", model_name, exc)
                self.unload()
                continue

        hint = (
            f" Only ~{free_gib:.1f} GiB free / {total_gib:.1f} GiB total before load."
            if free_gib is not None and total_gib is not None
            else ""
        )
        msg = (
            "CUDA OOM / load failure while loading the VLM."
            + hint
            + " Tried: "
            + ", ".join(ordered)
            + ". Kill other GPU processes (`nvidia-smi`), or pass "
            "--vlm-model Qwen/Qwen2-VL-2B-Instruct. "
            f"Original error: {last_exc}"
        )
        self._load_failed = True
        self._load_error = msg
        raise VLMExtractionError(msg) from last_exc

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
    # HTML table parsing (primary path)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_html_tables(text: str) -> List[str]:
        """Find every `<table ...>...</table>` fragment in the raw response, in order."""
        return re.findall(r"<table\b.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _html_table_to_grid(html_fragment: str) -> Optional[List[List[str]]]:
        """
        Parse one `<table>` fragment into a dense 2D grid of raw cell text,
        expanding `colspan`/`rowspan` generically (standard HTML-table-to-
        grid algorithm: cells spanning multiple rows are "carried forward"
        into a `pending` map until the row they land on is reached).

        Deliberately does NOT use `pandas.read_html`: pandas silently
        reinterprets numeric-looking cell text (e.g. "250.000" -> 250.0),
        which corrupts exactly the kind of formatted numbers these forms are
        full of. Every cell here stays a plain string, verbatim.
        """
        from lxml import html as lxml_html

        try:
            table_el = lxml_html.fromstring(html_fragment)
        except Exception as exc:
            logger.warning("lxml could not parse an HTML table fragment: %s", exc)
            return None

        row_elements = table_el.xpath(".//tr")
        if not row_elements:
            return None

        grid: List[List[str]] = []
        pending: dict[tuple[int, int], str] = {}
        max_cols = 0

        for row_index, tr in enumerate(row_elements):
            while len(grid) <= row_index:
                grid.append([])
            col_index = 0
            for cell in tr.xpath("./td|./th"):
                while (row_index, col_index) in pending:
                    grid[row_index].append(pending.pop((row_index, col_index)))
                    col_index += 1
                text = " ".join((cell.text_content() or "").split())
                try:
                    colspan = max(1, int(cell.get("colspan", 1)))
                except (TypeError, ValueError):
                    colspan = 1
                try:
                    rowspan = max(1, int(cell.get("rowspan", 1)))
                except (TypeError, ValueError):
                    rowspan = 1
                for span_col in range(colspan):
                    grid[row_index].append(text)
                    for span_row in range(1, rowspan):
                        pending[(row_index + span_row, col_index + span_col)] = text
                    col_index += 1
            while (row_index, col_index) in pending:
                grid[row_index].append(pending.pop((row_index, col_index)))
                col_index += 1
            max_cols = max(max_cols, len(grid[row_index]))

        if max_cols == 0:
            return None
        for row in grid:
            row.extend([""] * (max_cols - len(row)))
        return grid

    @classmethod
    def _html_table_to_dataframe(cls, html_fragment: str) -> Optional[pd.DataFrame]:
        """Grid (see `_html_table_to_grid`) -> DataFrame, first row as header."""
        grid = cls._html_table_to_grid(html_fragment)
        if not grid:
            return None
        headers, body = grid[0], grid[1:]
        return cls._grid_to_dataframe(headers, body)

    def _parse_html_response(self, raw_text: str) -> List[pd.DataFrame]:
        """Parse every `<table>` fragment in the response into DataFrames."""
        fragments = self._extract_html_tables(raw_text)
        dataframes: List[pd.DataFrame] = []
        for fragment in fragments:
            df = self._html_table_to_dataframe(fragment)
            if df is not None:
                dataframes.append(df)
        return dataframes

    # ------------------------------------------------------------------
    # Robust JSON parsing (fallback path, for models that ignore the HTML
    # instruction and answer in JSON anyway -- see `_JSON_FALLBACK_HINT`)
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

    def _parse_json_response(self, raw_text: str) -> Optional[Any]:
        """
        Parse the VLM's raw text into a Python object (list or dict).

        If the model hallucinates invalid JSON, logs a warning and returns
        `None` so the caller can skip this page gracefully.
        """
        candidate = self._strip_markdown_fences(raw_text)
        # Prefer outer array; if the model returned a single object, fall back
        # to the first {...} span instead.
        array_candidate = self._extract_json_array_substring(candidate)
        try:
            return json.loads(array_candidate)
        except (json.JSONDecodeError, TypeError):
            pass

        obj_start, obj_end = candidate.find("{"), candidate.rfind("}")
        if obj_start != -1 and obj_end > obj_start:
            try:
                return json.loads(candidate[obj_start : obj_end + 1])
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("JSON fallback parse also failed (%s).", exc)
                return None

        return None

    @staticmethod
    def _normalize_cell(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ").strip()

    @classmethod
    def _grid_to_dataframe(cls, headers: List[Any], rows: List[Any]) -> Optional[pd.DataFrame]:
        """Build a DataFrame from a headers list + list-of-lists body (grid form)."""
        if not isinstance(rows, list) or not rows:
            # Header-only / empty body still counts as a detected table scaffold.
            if headers:
                cols = [cls._normalize_cell(h) or f"Column_{i + 1}" for i, h in enumerate(headers)]
                return pd.DataFrame(columns=cls._dedupe_columns(cols))
            return None

        width = max(len(headers) if headers else 0, max((len(r) for r in rows if isinstance(r, list)), default=0))
        if width == 0:
            return None

        if headers and len(headers) == width:
            cols = [cls._normalize_cell(h) or f"Column_{i + 1}" for i, h in enumerate(headers)]
        elif headers:
            cols = [cls._normalize_cell(h) or f"Column_{i + 1}" for i, h in enumerate(headers)]
            while len(cols) < width:
                cols.append(f"Column_{len(cols) + 1}")
            cols = cols[:width]
        else:
            cols = [f"Column_{i + 1}" for i in range(width)]

        cols = cls._dedupe_columns(cols)
        normalized: List[List[str]] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            cells = [cls._normalize_cell(c) for c in row]
            if len(cells) < width:
                cells = cells + [""] * (width - len(cells))
            normalized.append(cells[:width])

        if not normalized and not headers:
            return None
        return pd.DataFrame(normalized, columns=cols)

    @staticmethod
    def _dedupe_columns(names: List[str]) -> List[str]:
        seen: dict[str, int] = {}
        out: List[str] = []
        for name in names:
            base = name or "Column"
            if base not in seen:
                seen[base] = 1
                out.append(base)
            else:
                seen[base] += 1
                out.append(f"{base}_{seen[base]}")
        return out

    @classmethod
    def _payload_to_dataframes(cls, data: Any) -> List[pd.DataFrame]:
        """
        Accept several generic JSON shapes the VLM may emit — none of them
        encode a document template; they only describe a visual grid:

          A) [ {"headers": [...], "rows": [[...], ...] }, ... ]
          B)   {"headers": [...], "rows": [[...], ...] }
          C) [ ["h1","h2"], ["r1c1","r1c2"], ... ]   # first row = header
          D) [ {"colA": "...", "colB": "..."}, ... ]  # list of row objects
          E) {"tables": [ <A/B/C/D>... ]}
        """
        if data is None:
            return []

        if isinstance(data, dict) and "tables" in data:
            frames: List[pd.DataFrame] = []
            for item in data["tables"]:
                frames.extend(cls._payload_to_dataframes(item))
            return frames

        if isinstance(data, dict) and ("rows" in data or "headers" in data):
            df = cls._grid_to_dataframe(list(data.get("headers") or []), list(data.get("rows") or []))
            return [df] if df is not None else []

        if isinstance(data, list) and data and all(isinstance(x, dict) and ("rows" in x or "headers" in x) for x in data):
            frames = []
            for item in data:
                frames.extend(cls._payload_to_dataframes(item))
            return frames

        if isinstance(data, list) and data and all(isinstance(x, list) for x in data):
            # Pure 2D grid: treat first row as headers when it looks non-numeric-heavy.
            headers, body = data[0], data[1:]
            df = cls._grid_to_dataframe(list(headers), body)
            return [df] if df is not None else []

        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            try:
                df = pd.DataFrame(data)
            except (ValueError, TypeError) as exc:
                logger.warning("Failed to build DataFrame from row-objects: %s", exc)
                return []
            return [df] if not df.empty else []

        logger.warning("Unrecognized VLM JSON shape (%s) -- skipping.", type(data).__name__)
        return []

    @staticmethod
    def _headers_compatible(prev_cols: List[str], next_cols: List[str]) -> bool:
        """
        Require same width AND enough header-token overlap before stitching.

        Same column count alone is too weak: two unrelated form tables can
        share width and get wrongly glued, destroying page/position form.
        """
        if len(prev_cols) != len(next_cols) or len(prev_cols) == 0:
            return False

        def _norm(name: object) -> str:
            return " ".join(str(name).lower().split())

        prev_n = [_norm(c) for c in prev_cols]
        next_n = [_norm(c) for c in next_cols]

        generic = re.compile(r"^(?:col(?:_\d+)?|column_\d+)$")
        prev_generic = all(generic.fullmatch(c or "") for c in prev_n)
        next_generic = all(generic.fullmatch(c or "") for c in next_n)
        # Both sides synthetic -> width match is enough (continuation without headers).
        if prev_generic and next_generic:
            return True
        # One side synthetic -> allow stitch (continuation page often drops header).
        if prev_generic or next_generic:
            return True

        matches = sum(1 for a, b in zip(prev_n, next_n) if a and a == b)
        # Also accept high token overlap when order drifts slightly.
        prev_tokens = {t for c in prev_n for t in c.split() if t}
        next_tokens = {t for c in next_n for t in c.split() if t}
        overlap = len(prev_tokens & next_tokens) / max(1, len(prev_tokens | next_tokens))
        return matches >= max(1, len(prev_n) // 2) or overlap >= 0.5

    @classmethod
    def _stitch_continuation_tables(cls, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Generic multi-page stitch: only concatenate when consecutive tables
        share width AND compatible headers. Prevents unrelated same-width
        tables from being glued together.
        """
        if len(tables) <= 1:
            return tables

        stitched: List[pd.DataFrame] = [tables[0].copy()]
        for nxt in tables[1:]:
            prev = stitched[-1]
            if cls._headers_compatible(list(prev.columns), list(nxt.columns)):
                cont = nxt.copy()
                cont.columns = list(prev.columns)
                stitched[-1] = pd.concat([prev, cont], ignore_index=True)
                logger.info(
                    "Stitched continuation table (%d + %d rows, %d cols).",
                    len(prev),
                    len(nxt),
                    len(prev.columns),
                )
            else:
                stitched.append(nxt.copy())
        return stitched

    # ------------------------------------------------------------------
    # Per-page + full-document extraction
    # ------------------------------------------------------------------
    def extract_tables_from_image(self, image: Image.Image) -> List[pd.DataFrame]:
        """
        Run the full single-page pipeline: inference -> parse -> DataFrame(s).

        Tries HTML `<table>` parsing first (the primary prompt format), then
        falls back to the generic JSON grid shapes for models/checkpoints
        that answer in JSON despite the instruction. Returns an empty list
        (never raises) if the page had no table or nothing could be parsed
        -- callers should treat that as "skip this page", not a fatal error.
        One page may yield multiple tables when several distinct grids are
        present.
        """
        try:
            raw_text = self._run_inference(image)
        except VLMExtractionError:
            # Load/OOM failures must abort the whole VLM pass (do not retry per page).
            raise
        except Exception as exc:  # pragma: no cover - defensive: GPU/runtime errors
            logger.warning("VLM inference failed on a page: %s", exc)
            return []

        html_tables = self._parse_html_response(raw_text)
        if html_tables:
            return html_tables

        payload = self._parse_json_response(raw_text)
        if payload is not None:
            json_tables = self._payload_to_dataframes(payload)
            if json_tables:
                return json_tables

        logger.warning(
            "VLM response had no parseable HTML or JSON table. Raw response (first 500 chars): %.500s",
            raw_text,
        )
        return []

    def extract_table_from_image(self, image: Image.Image) -> Optional[pd.DataFrame]:
        """Backward-compatible wrapper: first table on the page, or None."""
        tables = self.extract_tables_from_image(image)
        return tables[0] if tables else None

    def extract_pages(
        self,
        pdf_path: str | Path,
        page_indices: Optional[List[int]] = None,
        apply_ocr_cleanup: bool = True,
        *,
        crop_to_table: Optional[bool] = None,
    ) -> List[pd.DataFrame]:
        """
        Extract tables from a specific subset of pages (0-based indices), in memory.

        This is the method the content-driven orchestrator
        (`unified_pipeline.py`) calls: it first uses a cheap `pdfplumber`
        scan to find which pages physically contain a table, then passes
        ONLY those page indices here -- the VLM never runs on pages that are
        pure prose. Pass `page_indices=None` to process every page (used by
        the standalone `extract()` / CLI entrypoint below).

        Pages are rendered to IMAGES (the PDF text layer is never read for
        table cells). Consecutive page tables with the same column count are
        stitched into one DataFrame (generic continuation, no template).

        Does NOT write any file -- callers own export (so the orchestrator
        can combine results from multiple backends into one workbook).

        Args:
            crop_to_table: Override `VLMConfig.crop_to_table` for this call.
                Pass False to force full-page images (useful when the pixel
                crop heuristic clips a scanned form incorrectly).

        Returns:
            Cleaned DataFrame(s) for every usable table found (may be empty).
        """
        pdf_path = Path(pdf_path)
        use_crop = self.config.crop_to_table if crop_to_table is None else crop_to_table

        try:
            page_images = render_pdf_to_images(
                pdf_path,
                dpi=self.config.render_dpi,
                page_indices=page_indices,
                crop_to_table=use_crop,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise VLMExtractionError(f"Failed to render '{pdf_path}' to images: {exc}") from exc

        # Fail fast once if the model cannot load (avoid N pages × OOM retries).
        self.load()

        dataframes: List[pd.DataFrame] = []
        for image in page_images:
            for df in self.extract_tables_from_image(image):
                if df is None or df.empty:
                    continue
                if apply_ocr_cleanup:
                    df = clean_ocr_errors(df)
                dataframes.append(df)

        return self._stitch_continuation_tables(dataframes)

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
