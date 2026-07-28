# Build from repo root:
#   docker compose -f ocr_pipeline_backup/ocr_pipeline/docker-compose.yml build table_extract

FROM python:3.10-slim

WORKDIR /app

# OpenCV / PaddleOCR need libGL + X libs even on CPU servers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ocr_pipeline_backup/ocr_pipeline/table_extract/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pdf_extractor /app/pdf_extractor
COPY ocr_pipeline_backup/ocr_pipeline/table_extract/app.py /app/app.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TABLE_EXTRACT_BACKEND=paddle
# Avoid GUI backends inside container
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

CMD ["python", "-u", "app.py"]
