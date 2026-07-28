"""
Kafka worker: pdf-arrived -> pdf_extractor -> structured_tables.json + xlsx on MinIO.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import boto3
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BUCKET = os.getenv("MINIO_BUCKET", "data-lake")
TABLE_BACKEND = os.getenv("TABLE_EXTRACT_BACKEND", "paddle").strip().lower()


def init_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "Admin@123"),
    )


def run_extract(pdf_path: Path, work_dir: Path):
    from pdf_extractor.pipeline_bridge import write_structured_tables_json
    from pdf_extractor.unified_pipeline import UnifiedPDFPipeline

    skip_docling = TABLE_BACKEND in {"paddle", "paddle-vl"}
    force_paddle_vl = TABLE_BACKEND == "paddle-vl"
    pipe = UnifiedPDFPipeline(
        skip_marker=True,
        skip_docling=skip_docling,
        skip_paddle_vl=not force_paddle_vl,
        force_paddle_vl=force_paddle_vl,
    )
    result = pipe.run(
        pdf_path,
        work_dir,
        write_txt=False,
        write_xlsx=True,
        write_docx=False,
        skip_docling=skip_docling,
    )
    tables = list(result.tables or [])
    json_path = write_structured_tables_json(tables, work_dir / "structured_tables.json")
    return result, tables, json_path


def main():
    broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    s3 = init_s3()

    consumer = KafkaConsumer(
        "pdf-arrived",
        bootstrap_servers=[broker],
        group_id="table-extract-group",
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )

    logging.info("table_extract worker started (backend=%s)", TABLE_BACKEND)

    for msg in consumer:
        data = msg.value or {}
        document_id = data.get("document_id") or "unknown"
        bucket = data.get("bucket") or BUCKET
        pdf_key = data.get("pdf_path")
        total_pages = data.get("total_pages")

        if not pdf_key:
            logging.warning("pdf-arrived missing pdf_path: %s", data)
            continue

        tmp = Path(tempfile.mkdtemp(prefix="table_extract_"))
        try:
            local_pdf = tmp / Path(pdf_key).name
            logging.info("Downloading s3://%s/%s", bucket, pdf_key)
            s3.download_file(bucket, pdf_key, str(local_pdf))

            work_dir = tmp / "out"
            work_dir.mkdir(parents=True, exist_ok=True)
            result, tables, local_json = run_extract(local_pdf, work_dir)

            prefix = f"tables/{document_id}"
            tables_json_key = f"{prefix}/structured_tables.json"
            xlsx_key = f"{prefix}/output_tables.xlsx"

            s3.upload_file(str(local_json), bucket, tables_json_key)
            if result.xlsx_path and Path(result.xlsx_path).is_file():
                s3.upload_file(str(result.xlsx_path), bucket, xlsx_key)
            else:
                xlsx_key = None

            payload = {
                "document_id": document_id,
                "bucket": bucket,
                "tables_json_path": tables_json_key,
                "xlsx_path": xlsx_key,
                "status": "success",
                "table_count": len(tables),
                "total_pages": total_pages,
                "backend": result.table_backend_used,
                "finished_at": time.time(),
            }
            producer.send("table-extract-completed", payload)
            producer.flush()
            logging.info(
                "table-extract-completed %s tables=%d -> %s",
                document_id,
                len(tables),
                tables_json_key,
            )
        except Exception as exc:
            logging.exception("table_extract failed for %s: %s", document_id, exc)
            producer.send(
                "table-extract-completed",
                {
                    "document_id": document_id,
                    "bucket": bucket,
                    "tables_json_path": None,
                    "xlsx_path": None,
                    "status": "error",
                    "error": str(exc),
                    "table_count": 0,
                    "total_pages": total_pages,
                },
            )
            producer.flush()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
