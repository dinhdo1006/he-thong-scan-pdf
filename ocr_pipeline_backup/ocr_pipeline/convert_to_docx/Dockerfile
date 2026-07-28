FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY structured_docx.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "app.py"]
