FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

# Cài đặt PyTorch phiên bản CPU (nhẹ hơn và không cần driver GPU)
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY app.py .
COPY weights ./weights

CMD ["python", "-u", "app.py"]