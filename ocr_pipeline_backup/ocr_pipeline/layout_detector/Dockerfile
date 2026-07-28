FROM python:3.10-slim

WORKDIR /app

# 1. Bổ sung 'build-essential' để có gcc/g++ biên dịch các thư viện C
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 2. Nâng cấp pip lên bản mới nhất trước khi cài đặt requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# PATCH PaddleOCR bug dt_boxes
RUN sed -i 's/if not dt_boxes:/if dt_boxes is None or len(dt_boxes) == 0:/g' \
    /usr/local/lib/python3.10/site-packages/paddleocr/paddleocr.py

# Copy toàn bộ file code vào container
COPY . .

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

# Khởi chạy script 
CMD ["python", "-u", "app.py"]