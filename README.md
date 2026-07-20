# Pipeline extract PDF offline với Marker

Script Python **100% local** (không gọi API) để tách text và bảng từ PDF phức tạp (scan + native).

## Cài đặt

Yêu cầu: **Python 3.10+**

```bash
pip install marker-pdf
pip install pandas openpyxl
```

Hoặc:

```bash
pip install -r requirements.txt
```

Lần đầu chạy, Marker/Surya sẽ tải model về máy (cần mạng lúc setup). Sau đó xử lý offline.

GPU tùy chọn:

```bash
set TORCH_DEVICE=cuda
```

## Cách dùng

```bash
python extract_pdf.py --input path/to/file.pdf --output-dir ./output
```

Tùy chọn:

| Flag | Mô tả |
|------|--------|
| `-i` / `--input` | Đường dẫn PDF (bắt buộc) |
| `-o` / `--output-dir` | Thư mục output (mặc định `./output`) |
| `--no-markdown` | Không lưu file Markdown trung gian |
| `-v` / `--verbose` | Log chi tiết |

## Output

| File | Nội dung |
|------|----------|
| `output_tables.xlsx` | Mỗi bảng một sheet (`Table_1`, `Table_2`, …) |
| `output_text.txt` | Toàn bộ text không thuộc bảng |
| `output_markdown.md` | Markdown trung gian từ Marker (trừ khi `--no-markdown`) |

## Pipeline

1. **Extraction** — `marker-pdf` (`PdfConverter`, `use_llm=False`) → Markdown
2. **Parsing** — tách block bảng GFM (`\|` + `|---|`) khỏi text
3. **Tables** — `pandas` DataFrame → Excel (`openpyxl`)
4. **Text** — các dòng còn lại → `.txt` (giữ khoảng đoạn)

## Cấu trúc code

```
extract_pdf.py                 # CLI
pdf_extractor/
  __init__.py
  marker_extractor.py          # Marker PDF → Markdown
  markdown_parser.py           # Tách table / text
  exporters.py                 # Excel + text writers
  pipeline.py                  # Orchestration
requirements.txt
```

## Lưu ý

- Không bật `--use_llm` / Gemini — đảm bảo offline.
- Bảng Markdown “bẩn” (ô nhiều dòng, merged cell) có thể lệch cột.
- License Marker: GPL / weights có điều kiện thương mại — kiểm tra trước khi dùng production lớn.
