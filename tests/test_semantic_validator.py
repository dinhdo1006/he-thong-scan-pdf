"""Tests for form-agnostic STT outline + VND sum semantic validators."""

from __future__ import annotations

from pdf_extractor.semantic_validator import (
    annotate_dataframe_semantics,
    list_invalid_rows,
    parse_vnd_amount,
    validate_stt_outline,
    validate_sum_consistency,
)


def test_validate_stt_outline_all_valid() -> None:
    rows = [{"stt": s} for s in ["I", "1", "1.1", "1.1.1", "1.1.2", "1.2", "2", "II"]]
    out = validate_stt_outline(rows)
    assert all(r["stt_valid"] for r in out)
    assert all(r["stt_issue"] is None for r in out)


def test_validate_stt_outline_missing_1_1_6_from_cd02() -> None:
    """
    Real gap from CD_02_2011_021.pdf output: after 1.1.5 the next code is 1.1.7
    (1.1.6 missing). The row at 1.1.7 must be flagged.
    """
    # Outline codes extracted from D:\\output.txt for CD_02_2011_021.pdf
    cd02_stt = [
        "I",
        "1",
        "1.1",
        "1.1.2",  # also missing 1.1.1 — flagged separately
        "1.1.3",
        "1.1.4",
        "1.1.5",
        "1.1.7",  # <-- required detection target
        "1.1.8",
        "1.2",
        "1.2.1",
        "1.3",
        "1.3.1",
        "1.3.2",
        "2",
        "2.1",
        "2.2",
        "II",
        "III",
    ]
    rows = [{"stt": s} for s in cd02_stt]
    out = validate_stt_outline(rows)

    by_stt = {r["stt"]: r for r in out}
    assert by_stt["1.1.7"]["stt_valid"] is False
    assert by_stt["1.1.7"]["stt_issue"] is not None
    assert "1.1.6" in by_stt["1.1.7"]["stt_issue"]
    assert "1.1.7" in by_stt["1.1.7"]["stt_issue"]
    assert "missing" in by_stt["1.1.7"]["stt_issue"]

    # Also expect the earlier gap 1.1 -> 1.1.2 (missing 1.1.1)
    assert by_stt["1.1.2"]["stt_valid"] is False
    assert "1.1.1" in (by_stt["1.1.2"]["stt_issue"] or "")


def test_validate_sum_consistency_match() -> None:
    rows = [
        {"stt": "1", "col1": "1.000.000", "is_total": True},
        {"stt": "1.1", "col1": "400.000"},
        {"stt": "1.2", "col1": "600.000"},
    ]
    out = validate_sum_consistency(rows, amount_columns=["col1"])
    parent = next(r for r in out if r["stt"] == "1")
    assert parent["sum_valid"] is True
    assert parent["sum_diff"] == 0.0


def test_validate_sum_consistency_mismatch_reports_diff() -> None:
    rows = [
        {"stt": "1", "col1": "1.000.000"},
        {"stt": "1.1", "col1": "400.000"},
        {"stt": "1.2", "col1": "500.000"},  # sum children = 900_000, Δ = -100_000
    ]
    out = validate_sum_consistency(rows, amount_columns=["col1"])
    parent = next(r for r in out if r["stt"] == "1")
    assert parent["sum_valid"] is False
    assert parent["sum_diff"] == -100_000.0


def test_parse_vnd_amount_grouping_and_negative() -> None:
    assert parse_vnd_amount("10.620.000") == 10_620_000.0
    assert parse_vnd_amount("(250.000)") == -250_000.0
    assert parse_vnd_amount("") == 0.0


def test_annotate_dataframe_adds_columns() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "STT": ["1", "1.1", "1.2"],
            "Số tiền": ["1.000.000", "400.000", "600.000"],
        }
    )
    annotated = annotate_dataframe_semantics(df)
    assert "STT_valid" in annotated.columns
    assert "Sum_check" in annotated.columns
    assert list(annotated["STT"]) == ["1", "1.1", "1.2"]
    assert annotated.loc[0, "Sum_check"] == "OK"


def test_cd02_invalid_list_report(capsys) -> None:
    """Run validators on the real CD_02 STT sequence and print all invalids."""
    cd02_stt = [
        "I",
        "1",
        "1.1",
        "1.1.2",
        "1.1.3",
        "1.1.4",
        "1.1.5",
        "1.1.7",
        "1.1.8",
        "1.2",
        "1.2.1",
        "1.3",
        "1.3.1",
        "1.3.2",
        "2",
        "2.1",
        "2.2",
        "II",
        "III",
    ]
    # Amounts approximated from D:\\output.txt parent/child money cells.
    amounts = {
        "I": "10.620.000",
        "1": "10.620.000",
        "1.1": "10.620.000",
        "1.1.2": "250.000",
    }
    rows = [{"stt": s, "amount": amounts.get(s, "")} for s in cd02_stt]
    rows = validate_stt_outline(rows)
    rows = validate_sum_consistency(rows, amount_columns=["amount"])
    invalids = list_invalid_rows(rows)

    print("=== CD_02_2011_021 semantic invalids ===")
    for item in invalids:
        print(f"  [{item['kind']}] stt={item['stt']!r} | {item['issue']}")
    print(f"=== total invalid events: {len(invalids)} ===")

    # Must include the 1.1.6 gap.
    assert any(
        i["stt"] == "1.1.7" and i["kind"] == "stt" and "1.1.6" in (i["issue"] or "")
        for i in invalids
    )
    captured = capsys.readouterr()
    assert "1.1.7" in captured.out
