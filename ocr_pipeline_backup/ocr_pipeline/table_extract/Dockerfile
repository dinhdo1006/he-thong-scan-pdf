# Build from repo root:
#   docker compose -f ocr_pipeline_backup/ocr_pipeline/docker-compose.yml build table_extract
#
# Install PaddleOCR / Docling on the host or extend this image for GPU workloads.
# Stage-1 default: mount pdf_extractor + run with host Python if image is too heavy.

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ocr_pipeline_backup/ocr_pipeline/table_extract/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Core package (table logic). Extra ML deps (paddleocr, etc.) installed separately.
COPY pdf_extractor /app/pdf_extractor
COPY ocr_pipeline_backup/ocr_pipeline/table_extract/app.py /app/app.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TABLE_EXTRACT_BACKEND=paddle

CMD ["python", "-u", "app.py"]
