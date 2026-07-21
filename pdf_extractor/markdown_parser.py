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

    def _table_lines_to_dataframe(
        self, table_lines: list[str]
    ) -> pd.DataFrame | None:
        """
        Convert a list of Markdown table lines into a pandas DataFrame.

        First non-separator row → header.
        Separator rows are discarded.
        Remaining rows → data.
        """
        if not table_lines:
            return None

        data_rows: list[list[str]] = []
        header: list[str] | None = None

        for line in table_lines:
            if self._is_separator_row(line):
                continue
            cells = self._split_row(line)
            if header is None:
                header = cells
            else:
                data_rows.append(cells)

        if header is None:
            return None

        # Normalize column count across rows (pad / truncate)
        n_cols = len(header)
        normalized: list[list[str]] = []
        for row in data_rows:
            if len(row) < n_cols:
                row = row + [""] * (n_cols - len(row))
            elif len(row) > n_cols:
                row = row[:n_cols]
            normalized.append(row)

        # Ensure unique column names for Excel safety
        seen: dict[str, int] = {}
        unique_header: list[str] = []
        for col in header:
            name = col if col else "col"
            if name in seen:
                seen[name] += 1
                unique_header.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_header.append(name)

        return pd.DataFrame(normalized, columns=unique_header)
