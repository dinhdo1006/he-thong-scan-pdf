# OCR Pipeline + pdf_extractor (table branch)

## Mục tiêu

- **Text:** Preprocess → layout → VietOCR → reconstruct  
- **Bảng:** worker `table_extract` (logic `pdf_extractor`) từ **PDF gốc**  
- **Đầu ra:** một file DOCX (`docx/{document_id}.docx`)

Không hardcode form B06/CD_*; bảng generic qua `header_matrix`.

## Topics Kafka

| Topic | Producer | Consumer |
|---|---|---|
| `pdf-arrived` | NiFi / script | `table_extract` |
| `image-arrived` | NiFi | `layout_detector` |
| `crop-completed` | layout | vietocr |
| `ocr-completed` | vietocr | reconstruct |
| `document-completed` | reconstruct | merge |
| `table-extract-completed` | table_extract | merge |
| `merge-completed` | merge | convert_to_docx |
| `pipeline-completed` | convert_to_docx | (downstream) |

### `pdf-arrived`

```json
{
  "document_id": "DOC_001",
  "bucket": "data-lake",
  "pdf_path": "raw/DOC_001.pdf",
  "total_pages": 3
}
```

### `table-extract-completed`

```json
{
  "document_id": "DOC_001",
  "bucket": "data-lake",
  "tables_json_path": "tables/DOC_001/structured_tables.json",
  "xlsx_path": "tables/DOC_001/output_tables.xlsx",
  "status": "success",
  "table_count": 1,
  "total_pages": 3
}
```

### `merge-completed`

Thêm field `tables_json_path` (có thể null nếu timeout / error).

## MinIO layout

- `raw/{id}.pdf` — PDF gốc  
- `pic/...` — ảnh trang  
- `preprocessed/.../page_N.json` — text reconstruct  
- `tables/{id}/structured_tables.json` — bảng structured  
- `tables/{id}/output_tables.xlsx` — debug Excel  
- `doc_json/{id}.json` — merge pages  
- `docx/{id}.docx` — **output cuối**

## Env quan trọng

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `TABLE_EXTRACT_BACKEND` | `paddle` | Backend bảng |
| `MERGE_REQUIRE_TABLES` | `1` | Đợi `table-extract-completed` |
| `MERGE_TIMEOUT_S` | `600` | Timeout rồi merge fallback |
| `ENABLE_ROBOFLOW_TABLES` | `0` | Tắt Roboflow (bảng dùng `table_extract`) |
| `ROBOFLOW_API_KEY` | (rỗng) | Chỉ khi bật Roboflow |

## Chạy local smoke (không cần Kafka)

Từ repo root:

```bash
python ocr_pipeline_backup/ocr_pipeline/scripts/smoke_structured_docx.py
python ocr_pipeline_backup/ocr_pipeline/scripts/smoke_structured_docx.py --pdf "path/to.pdf" -o ./smoke_out
```

## Trigger thủ công (Kafka)

1. Upload PDF lên `raw/{id}.pdf`  
2. Publish `pdf-arrived`  
3. (Nhánh text) preprocess + `image-arrived` như cũ  
4. Đợi `pipeline-completed` và tải `docx/{id}.docx`

## NiFi (giai đoạn sau)

NiFi cần: PutObject PDF → Produce `pdf-arrived`; PutObject page images → `image-arrived`; truyền `document_id` + `total_pages` thống nhất.

## Build note

`table_extract` image chỉ đóng gói `pdf_extractor` + Kafka/MinIO deps. Cần cài thêm PaddleOCR/Docling trên image hoặc chạy worker trên host GPU với `PYTHONPATH` trỏ repo.
