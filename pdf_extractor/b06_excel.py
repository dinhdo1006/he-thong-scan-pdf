"""Compatibility shim — prefer ``hierarchy_excel`` (form-agnostic)."""

from .hierarchy_excel import (  # noqa: F401
    drop_semantic_helpers,
    write_hierarchy_workbook,
)

# Legacy aliases used by older call sites / tests.
write_b06_workbook = write_hierarchy_workbook


def is_b06_excel_candidate(df) -> bool:  # type: ignore[no-untyped-def]
    """Deprecated: always True so exporter uses hierarchy writer for all tables."""
    return df is not None and len(getattr(df, "columns", [])) > 0
