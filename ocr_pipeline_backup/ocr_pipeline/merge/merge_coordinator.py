"""
Merge coordinator: wait for text pages + structured table extract before emitting.

Pure logic (no Kafka/MinIO) so it can be unit-tested and reused by OCR workers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMergeState:
    document_id: str
    total_pages: int | None = None
    page_json_paths: dict[int, str] = field(default_factory=dict)
    tables_json_path: str | None = None
    table_status: str | None = None
    first_seen: float = field(default_factory=time.time)
    text_folder: str | None = None

    def page_count(self) -> int:
        return len(self.page_json_paths)

    def pages_ready(self) -> bool:
        if self.total_pages is None:
            return False
        return self.page_count() >= int(self.total_pages)

    def tables_ready(self) -> bool:
        return bool(self.tables_json_path) or self.table_status in {
            "success",
            "error",
            "skipped",
            "timeout",
        }

    def is_complete(self, *, require_tables: bool = True) -> bool:
        if not self.pages_ready():
            return False
        if require_tables and not self.tables_ready():
            return False
        return True

    def timed_out(self, timeout_s: float) -> bool:
        return (time.time() - self.first_seen) >= timeout_s

    def can_emit_on_timeout(self) -> bool:
        """Allow partial merge after timeout if at least one page exists."""
        return self.page_count() >= 1



class MergeCoordinator:
    """Buffer per document_id until pages (+ optional tables) are ready."""

    def __init__(
        self,
        *,
        require_tables: bool = True,
        timeout_s: float = 600.0,
    ) -> None:
        self.require_tables = require_tables
        self.timeout_s = timeout_s
        self._docs: dict[str, DocumentMergeState] = {}

    def _state(self, document_id: str) -> DocumentMergeState:
        if document_id not in self._docs:
            self._docs[document_id] = DocumentMergeState(document_id=document_id)
        return self._docs[document_id]

    def on_page(
        self,
        *,
        document_id: str,
        json_path: str,
        page_name: str | None = None,
        total_pages: int | None = None,
        page_number: int | None = None,
    ) -> DocumentMergeState | None:
        """Record one reconstructed page. Returns state if ready to merge."""
        st = self._state(document_id)
        if total_pages is not None:
            st.total_pages = int(total_pages)
        if st.text_folder is None and json_path:
            st.text_folder = json_path.rsplit("/", 1)[0] if "/" in json_path else ""

        idx = page_number
        if idx is None and page_name:
            digits = "".join(ch for ch in page_name if ch.isdigit())
            idx = int(digits) if digits else st.page_count() + 1
        if idx is None:
            idx = st.page_count() + 1
        st.page_json_paths[int(idx)] = json_path
        return self._maybe_ready(st)

    def on_tables(
        self,
        *,
        document_id: str,
        tables_json_path: str | None,
        status: str = "success",
        total_pages: int | None = None,
    ) -> DocumentMergeState | None:
        st = self._state(document_id)
        if total_pages is not None:
            st.total_pages = int(total_pages)
        st.tables_json_path = tables_json_path
        st.table_status = status
        return self._maybe_ready(st)

    def poll_timeouts(self) -> list[DocumentMergeState]:
        ready: list[DocumentMergeState] = []
        for st in list(self._docs.values()):
            if st.timed_out(self.timeout_s) and st.can_emit_on_timeout():
                if not st.tables_ready():
                    st.table_status = "timeout"
                ready.append(st)
        return ready

    def pop(self, document_id: str) -> DocumentMergeState | None:
        return self._docs.pop(document_id, None)

    def _maybe_ready(self, st: DocumentMergeState) -> DocumentMergeState | None:
        if st.is_complete(require_tables=self.require_tables):
            return st
        if st.timed_out(self.timeout_s) and st.can_emit_on_timeout():
            if self.require_tables and not st.tables_ready():
                st.table_status = "timeout"
            return st
        return None

    def merge_payload(self, st: DocumentMergeState) -> dict[str, Any]:
        pages = [st.page_json_paths[k] for k in sorted(st.page_json_paths)]
        return {
            "document_id": st.document_id,
            "page_json_paths": pages,
            "merged_pages_count": len(pages),
            "tables_json_path": st.tables_json_path,
            "table_status": st.table_status,
            "total_pages": st.total_pages,
            "text_folder": st.text_folder,
        }
