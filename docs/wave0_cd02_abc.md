# Wave 0 — Phân loại lỗi A/B/C trên CD_02_2011_021

**PDF:** `D:\pdf test\CD_02_2011_021.pdf`  
**Baseline output:** `D:\output.txt` / `D:\output_ocr\output_tables.xlsx` (Paddle/Docling cascade, skip Marker)  
**Ngày đo:** 2026-07-27

## Định nghĩa nhóm

| Nhóm | Ý nghĩa | Lớp code chịu trách nhiệm |
|------|---------|---------------------------|
| **A** | Sai **nội dung ô** (OCR / nhận dạng chữ) | Docling / PP-Structure / VL |
| **B** | Đúng vùng nhưng **sai cột/hàng** (layout lưới) | TableFormer / flatten / stitch / repair |
| **C** | Đúng bảng nhưng **sai chỗ trong document** (thứ tự đọc) | `assemble` / `text_fallback` / merge Marker |

## Kết quả đo CD_02

| # | Hiện tượng quan sát | Nhóm | Ghi chú |
|---|---------------------|------|---------|
| 1 | Chữ ký / chức danh (`Nguyễn Xuân Hương`, `PHÓ CỤC TRƯỞNG`) nằm **trước** bảng thay vì sau | **C** | `_finalize_prose` gộp header+footer rồi bảng append cuối — gốc bệnh Wave 1 |
| 2 | Header form (`Mẫu số`, `Đơn vị`) đúng vùng đầu file | OK | Prose above-bbox hoạt động |
| 3 | Nhảy STT `1.1.5` → `1.1.7` (thiếu `1.1.6`) | **A** (hoặc A+B) | OCR/backend bỏ dòng; cần VL/semantic so sánh |
| 4 | Dòng trống xen kẽ trong lưới | **B** | Layout/span; `repair_form_table` chưa đủ |
| 5 | Lỗi chính tả OCR: `nuróc`, `vièn`, `giá tri`, `phai` | **A** | Nội dung ô; cleanup/OCR engine |
| 6 | Cột tiền VND `10.620.000` / `250.000` đúng hàng I / 1 / 1.1 / 1.1.2 | OK (một phần) | Backend bóc số khá ổn |
| 7 | Không còn pipe-table Marker làm neo vị trí | **C** | Skip Marker → cần assemble theo page bbox (Wave 1.1) |
| 8 | `merge_table_preserving_form` với skeleton rác không áp dụng (0 Marker tables) | **C** rủi ro | Khi bật Marker lại, header garbled từng gây lệch — Wave 1.2 |

## Quyết định sau Wave 0

1. **Wave 1 bắt buộc** — lỗi C (#1, #7) giải thích cảm giác “ghép vị trí form sai”.
2. **Wave 2 (PaddleOCR-VL Tier 1)** — nhắm lỗi A (#3, #5) và cạnh tranh semantic với Docling/PP-Structure; không thay thế việc sửa assemble.
3. Lỗi B (#4) giữ `repair_form_table` + token→cell (Wave 1.3) làm lớp phụ.

## Done criteria (Wave 0)

- [x] Có bảng phân loại A/B/C trên ≥1 form thật (CD_02).
- [x] Xác nhận: VL không phải ưu tiên đầu — sửa ghép (C) trước.
