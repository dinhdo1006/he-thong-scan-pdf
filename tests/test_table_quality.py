"""Tests for generic table quality scoring and spatial bbox assignment."""

from __future__ import annotations

import pandas as pd
import pytest

from pdf_extractor.table_quality import (
    BBox,
    TableCellBox,
    TextToken,
    assign_tokens_to_cells,
    grid_to_dataframe,
    pick_better_table,
    score_table_quality,
)


def test_score_penalizes_garbled_headers() -> None:
    dirty = pd.DataFrame(
        [["1", "x"]],
        columns=["A", "foo |_bar |_baz " + ("x" * 50)],
    )
    clean = pd.DataFrame([["1", "x"]], columns=["STT", "Mo_ta"])
    assert score_table_quality(clean) > score_table_quality(dirty)


def test_pick_better_table_prefers_cleaner() -> None:
    dirty = pd.DataFrame([["1"]], columns=["a|_b|_c " + ("z" * 80)])
    clean = pd.DataFrame([["1", "ok"]], columns=["A", "B"])
    chosen, winner = pick_better_table(dirty, clean)
    assert winner == "right"
    assert chosen.equals(clean)


def test_assign_tokens_to_cells_by_containment() -> None:
    cells = [
        TableCellBox(0, 0, BBox(0, 0, 10, 10)),
        TableCellBox(0, 1, BBox(10, 0, 20, 10)),
        TableCellBox(1, 0, BBox(0, 10, 10, 20)),
        TableCellBox(1, 1, BBox(10, 10, 20, 20)),
    ]
    tokens = [
        TextToken("STT", BBox(1, 1, 4, 4)),
        TextToken("Ten", BBox(12, 1, 18, 4)),
        TextToken("1", BBox(2, 12, 5, 15)),
        TextToken("An", BBox(12, 12, 16, 15)),
    ]
    grid = assign_tokens_to_cells(tokens, cells)
    assert grid == [["STT", "Ten"], ["1", "An"]]


def test_assign_tokens_keeps_anchor_only_for_colspan() -> None:
    # Wide cell spanning two columns -- text only at anchor (0,0).
    cells = [TableCellBox(0, 0, BBox(0, 0, 20, 10), col_span=2)]
    tokens = [TextToken("Group", BBox(5, 2, 8, 5))]
    grid = assign_tokens_to_cells(tokens, cells, n_rows=1, n_cols=2)
    assert grid[0][0] == "Group"
    assert grid[0][1] == ""


def test_grid_to_dataframe_uses_first_row_as_header() -> None:
    df = grid_to_dataframe([["A", "B"], ["1", "2"]])
    assert list(df.columns) == ["A", "B"]
    assert df.iloc[0].tolist() == ["1", "2"]


def test_pick_better_table_requires_one_side() -> None:
    with pytest.raises(ValueError):
        pick_better_table(None, None)
