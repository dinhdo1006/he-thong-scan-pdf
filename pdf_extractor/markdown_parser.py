"""Parse Marker Markdown into table blocks and remaining plain text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table detection regexes (GFM / pipe tables)
# ---------------------------------------------------------------------------
#
# SEPARATOR_RE matches a Markdown table separator row such as:
#   |---|---|
#   | --- | :---: | ---: |
#   ---|---
#
# Breakdown:
#   ^\s*\|?          optional leading pipe + whitespace
#   \s*:?-{3,}:?\s*  first column dashes (min 3), optional alignment colons
#   ( \| \s*:?-{3,}:?\s* )+   one or more additional "| ---" column groups
#   \|?\s*$          optional trailing pipe
#
# Minimum 2 dashes per column segment (Marker sometimes emits `--` not `---`).
SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$"
)

# A candidate table row must contain at least one pipe character.
PIPE_RE = re.compile(r"\|")


@dataclass
class ParseResult:
    """Result of splitting Markdown into tables and non-table text."""

    tables: list[pd.DataFrame] = field(default_factory=list)
    text: str = ""


@dataclass
class DocumentBlock:
    """One ordered slice of a Marker Markdown document."""

    kind: Literal["text", "table"]
    content: str | pd.DataFrame


class MarkdownBlockParser:
    """
    Line-by-line Markdown parser that separates pipe tables from prose.

    A table block starts only when a pipe-row is immediately followed by
    (or is itself) a separator matching SEPARATOR_RE. This avoids treating
    prose lines that happen to contain a single '|' as tables.
    """

    def parse_blocks(self, markdown: str) -> list[DocumentBlock]:
        """
        Walk Markdown and return an ORDERED list of text / table blocks.

        Preserves the original reading order (prose, then table, then more
        prose, ...) so callers can reassemble one plain-text file without
        losing content when a downstream table extractor fails.
        """
        lines = markdown.splitlines()
        blocks: list[DocumentBlock] = []
        text_lines: list[str] = []

        def flush_text() -> None:
            if not text_lines:
                return
            text = "\n".join(text_lines).rstrip()
            if text:
                blocks.append(DocumentBlock(kind="text", content=text))
            text_lines.clear()

        i = 0
        n = len(lines)
        while i < n:
            if self._is_table_start(lines, i):
                flush_text()
                table_lines, next_i = self._consume_table_block(lines, i)
                df = self._table_lines_to_dataframe(table_lines)
                if df is not None and not df.empty:
                    blocks.append(DocumentBlock(kind="table", content=df))
                    logger.debug("Block table_%d shape %s", sum(1 for b in blocks if b.kind == "table"), df.shape)
                else:
                    text_lines.extend(table_lines)
                i = next_i
                continue
            text_lines.append(lines[i])
            i += 1

        flush_text()
        logger.info(
            "Parsed %d block(s): %d table(s), %d text block(s).",
            len(blocks),
            sum(1 for b in blocks if b.kind == "table"),
            sum(1 for b in blocks if b.kind == "text"),
        )
        return blocks

    def parse(self, markdown: str) -> ParseResult:
        """
        Walk the Markdown string line by line and split table / text blocks.

        Args:
            markdown: Full document Markdown from Marker.

        Returns:
            ParseResult with DataFrames and remaining text (paragraph spacing kept).
        """
        blocks = self.parse_blocks(markdown)
        tables = [b.content for b in blocks if b.kind == "table"]
        text_parts = [b.content for b in blocks if b.kind == "text"]
        text = "\n\n".join(text_parts)
        if text_parts:
            text = text.rstrip() + "\n"
        return ParseResult(tables=tables, text=text)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _is_table_start(self, lines: list[str], index: int) -> bool:
        """
        Return True if `lines[index]` begins a valid Markdown pipe table.

        Requires a pipe character on the header (or body) row AND a
        separator row matching SEPARATOR_RE on the current or next line.
        """
        line = lines[index]
        if not PIPE_RE.search(line):
            return False

        # Case: current line IS the separator (malformed but possible)
        if SEPARATOR_RE.match(line):
            return True

        # Case: classic GFM — header row, then separator on next line
        if index + 1 < len(lines) and SEPARATOR_RE.match(lines[index + 1]):
            return True

        return False

    def _is_table_row(self, line: str) -> bool:
        """
        A continuation table row is any non-empty line that still contains '|'.

        Blank lines end the table block (GFM does not allow blank rows inside
        a pipe table).
        """
        if not line.strip():
            return False
        return bool(PIPE_RE.search(line))

    def _consume_table_block(
        self, lines: list[str], start: int
    ) -> tuple[list[str], int]:
        """
        Collect consecutive table rows starting at `start`.

        Returns:
            (table_lines, index_after_block)
        """
        block: list[str] = []
        i = start
        while i < len(lines) and self._is_table_row(lines[i]):
            block.append(lines[i])
            i += 1
        return block, i

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _split_row(self, line: str) -> list[str]:
        """
        Split a pipe-table row into cell strings.

        Leading/trailing empty cells created by border pipes are dropped:
          '| A | B |'  →  ['A', 'B']
          'A | B'      →  ['A', 'B']
        """
        # Strip outer whitespace, then split on '|'
        raw = line.strip()
        # Remove a single leading/trailing pipe used as border
        if raw.startswith("|"):
            raw = raw[1:]
        if raw.endswith("|"):
            raw = raw[:-1]
        cells = [c.strip() for c in raw.split("|")]
        return cells

    def _is_separator_row(self, line: str) -> bool:
        return bool(SEPARATOR_RE.match(line))

    # ------------------------------------------------------------------
    # Header-merging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_mostly_empty_row(cells: list[str], empty_ratio: float = 0.6, n_cols: int = 0) -> bool:
        """True when ≥ empty_ratio of cells (up to n_cols) are blank."""
        width = n_cols if n_cols > 0 else len(cells)
        if width == 0:
            return True
        padded = (cells + [""] * width)[:width]
        empty = sum(1 for c in padded if not c.strip())
        return empty / width >= empty_ratio

    @staticmethod
    def _merge_header_rows(header_rows: list[list[str]], n_cols: int = 0) -> list[str]:
        """
        Collapse multiple header rows into one by joining non-empty parts
        per column with a space. Cleans up <br> tags, extra whitespace.
        """
        if not header_rows:
            return []
        width = n_cols if n_cols > 0 else max(len(r) for r in header_rows)
        merged: list[str] = []
        for col_idx in range(width):
            parts: list[str] = []
            for row in header_rows:
                cell = row[col_idx] if col_idx < len(row) else ""
                cell = cell.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ").strip()
                if cell:
                    parts.append(cell)
            merged.append(" ".join(parts))
        return merged

    def _table_lines_to_dataframe(
        self, table_lines: list[str]
    ) -> pd.DataFrame | None:
        """
        Convert a list of Markdown table lines into a pandas DataFrame.

        Strategy for Marker's complex multi-row headers:
        1. The row(s) BEFORE the `---` separator are merged column-by-column
           into a single header (join non-empty parts with a space).
        2. After the separator, any leading rows where ≥60% of cells are
           empty are treated as sub-header rows and also merged into the
           header (Marker puts sub-header continuation rows in the data area).
        3. The first "real" data row is the first post-separator row that
           has <60% empty cells.
        4. Column names are always made unique and never blank.
        """
        if not table_lines:
            return None

        pre_sep: list[list[str]] = []
        post_sep: list[list[str]] = []
        found_sep = False

        for line in table_lines:
            if self._is_separator_row(line):
                found_sep = True
                continue
            cells = self._split_row(line)
            if not found_sep:
                pre_sep.append(cells)
            else:
                post_sep.append(cells)

        if not pre_sep and not post_sep:
            return None

        # Determine column width from the widest row seen
        all_rows = pre_sep + post_sep
        n_cols = max((len(r) for r in all_rows), default=0)
        if n_cols == 0:
            return None

        # Absorb leading mostly-empty post-sep rows into the header group
        extra_header_rows: list[list[str]] = []
        data_rows: list[list[str]] = []
        real_started = False
        for row in post_sep:
            if not real_started and self._is_mostly_empty_row(row, n_cols=n_cols):
                extra_header_rows.append(row)
            else:
                real_started = True
                data_rows.append(row)

        all_header_rows = pre_sep + extra_header_rows
        header = self._merge_header_rows(all_header_rows, n_cols=n_cols)

        if not header:
            return None

        # Normalize data row widths
        normalized: list[list[str]] = []
        for row in data_rows:
            padded = list(row) + [""] * (n_cols - len(row))
            normalized.append(padded[:n_cols])

        # Ensure unique, non-blank column names
        seen: dict[str, int] = {}
        unique_cols: list[str] = []
        for col in header:
            name = col.strip() if col.strip() else "col"
            if name not in seen:
                seen[name] = 0
                unique_cols.append(name)
            else:
                seen[name] += 1
                unique_cols.append(f"{name}_{seen[name]}")

        return pd.DataFrame(normalized, columns=unique_cols)
