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


def test_score_breakdown_matches_scalar_and_explains_zero_columns() -> None:
    """return_breakdown must not change scoring math; only annotate components."""
    clean = pd.DataFrame([["1", "x"]], columns=["STT", "Mo_ta"])
    scalar = score_table_quality(clean)
    scored, breakdown = score_table_quality(clean, return_breakdown=True)
    assert scored == scalar
    assert breakdown["score"] == scored
    assert breakdown["early_exit"] is None
    assert breakdown["n_cols"] == 2

    empty = pd.DataFrame()
    zero, empty_bd = score_table_quality(empty, return_breakdown=True)
    assert zero == 0.0
    assert empty_bd["early_exit"] == "none_or_zero_columns"


def test_pick_better_table_prefers_cleaner() -> None:
    dirty = pd.DataFrame([["1"]], columns=["a|_b|_c " + ("z" * 80)])
    clean = pd.DataFrame([["1", "ok"]], columns=["A", "B"])
    chosen, winner = pick_better_table(dirty, clean)
    assert winner == "right"
    assert chosen.equals(clean)


def test_pick_better_table_prefers_wider_over_narrow_high_score() -> None:
    """A clean 3-col fragment must not beat a usable 9-col table (stitch)."""
    narrow = pd.DataFrame(
        [["1.3.2", "Thu tai san", "x"]],
        columns=["STT", "Mo_ta", "col3"],
    )
    wide = pd.DataFrame(
        [["1.3.2", "Thu tai san", "", "", "", "", "", "", ""]],
        columns=[f"c{i}" for i in range(9)],
    )
    # Narrow often scores higher on fill-rate; width rule must still pick wide.
    assert score_table_quality(narrow) > score_table_quality(wide)
    chosen, winner = pick_better_table(narrow, wide)
    assert winner == "right"
    assert len(chosen.columns) == 9


def test_pick_better_table_allows_narrow_if_wide_collapsed() -> None:
    wide_garbage = pd.DataFrame(
        [[""] * 9],
        columns=["a|_b|_c " + ("z" * 80)] + [f"c{i}" for i in range(8)],
    )
    narrow_clean = pd.DataFrame([["1", "ok"]], columns=["A", "B"])
    # Wide score near-zero -> width preference does not apply; score picks clean.
    chosen, winner = pick_better_table(wide_garbage, narrow_clean)
    assert winner == "right"
    assert chosen.equals(narrow_clean)


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


def test_pick_better_table_semantic_overrides_higher_geometric_score(tmp_path) -> None:
    """
    Candidate with cleaner outline wins even if geometric score is lower.

    Left: high fill / clean headers but missing 1.1.6 (gap 1.1.5 -> 1.1.7).
    Right: slightly messier headers but contiguous outline — fewer semantic fails.
    """
    # Left: good-looking headers, but outline gap (2 semantic stt fails if also 1.1->1.1.2).
    left = pd.DataFrame(
        {
            "STT": ["1", "1.1", "1.1.1", "1.1.5", "1.1.7"],
            "Mo_ta": ["a", "b", "c", "d", "e"],
            "So_tien": ["100", "100", "40", "30", "30"],
        }
    )
    # Right: contiguous outline through 1.1.7 — 0 outline fails.
    right = pd.DataFrame(
        {
            "STT": ["1", "1.1", "1.1.1", "1.1.2", "1.1.3", "1.1.4", "1.1.5", "1.1.6", "1.1.7"],
            "Mo_ta": ["a"] * 9,
            "So_tien": ["10"] * 9,
        }
    )
    # Force a case where left would win geometrically on fill if semantic ignored:
    # (left is denser). Semantic must still pick right.
    chosen, winner = pick_better_table(
        left,
        right,
        use_semantic=True,
        left_label="docling",
        right_label="paddle",
        conflict_dir=tmp_path,
        page_no=0,
    )
    assert winner == "right"
    assert chosen.equals(right)


def test_pick_better_table_writes_conflict_csv_when_both_fail_same_stt(tmp_path) -> None:
    left = pd.DataFrame(
        {
            "STT": ["1", "1.1", "1.1.5", "1.1.7"],
            "So_tien": ["1000", "1000", "0", "0"],
        }
    )
    right = pd.DataFrame(
        {
            "STT": ["1", "1.1", "1.1.5", "1.1.7"],
            "So_tien": ["900", "900", "0", "0"],
        }
    )
    _chosen, _winner = pick_better_table(
        left,
        right,
        use_semantic=True,
        left_label="docling",
        right_label="paddle",
        conflict_dir=tmp_path,
        page_no=0,
    )
    csvs = list(tmp_path.glob("conflict_*.csv"))
    assert len(csvs) == 1
    body = csvs[0].read_text(encoding="utf-8-sig")
    assert "CONFLICT_NEEDS_MANUAL_CHECK" in body
    assert "1.1.7" in body


def test_pick_better_table_use_semantic_false_keeps_geometric_only() -> None:
    # Narrow clean vs wide sparse: geometric width prefers wide (right).
    narrow = pd.DataFrame(
        [["1.3.2", "Thu tai san", "x"]],
        columns=["STT", "Mo_ta", "col3"],
    )
    wide = pd.DataFrame(
        [["1.3.2", "Thu tai san", "", "", "", "", "", "", ""]],
        columns=[f"c{i}" for i in range(9)],
    )
    chosen, winner = pick_better_table(narrow, wide, use_semantic=False)
    assert winner == "right"
    assert len(chosen.columns) == 9
