import os
import json
import logging
from io import BytesIO

import boto3
from kafka import KafkaConsumer, KafkaProducer

from structured_docx import create_document_from_pages_and_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BUCKET = os.getenv("MINIO_BUCKET", "data-lake")


def init_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "Admin@123"),
    )


def main():
    broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    s3 = init_s3()

    consumer = KafkaConsumer(
        "merge-completed",
        bootstrap_servers=[broker],
        group_id="docx-export-group",
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )

    logging.info("DOCX Exporter Worker started (structured tables supported)...")

    for msg in consumer:
        try:
            data = msg.value
            json_path = data.get("json_path")
            document_id = data.get("document_id", "unknown")
            tables_json_path = data.get("tables_json_path")

            if not json_path:
                continue

            logging.info("Received JSON for DOCX export: %s", json_path)

            obj = s3.get_object(Bucket=BUCKET, Key=json_path)
            json_data = json.loads(obj["Body"].read().decode("utf-8"))
            if not isinstance(json_data, list):
                logging.error("Expected list of pages in %s", json_path)
                continue

            structured_payload = None
            if tables_json_path:
                try:
                    tobj = s3.get_object(Bucket=BUCKET, Key=tables_json_path)
                    structured_payload = json.loads(tobj["Body"].read().decode("utf-8"))
                    logging.info("Loaded structured tables from %s", tables_json_path)
                except Exception as exc:
                    logging.warning(
                        "Failed to load structured tables %s (%s) -- using crude fallback",
                        tables_json_path,
                        exc,
                    )
                    structured_payload = None

            doc = create_document_from_pages_and_tables(json_data, structured_payload)

            docx_io = BytesIO()
            doc.save(docx_io)
            docx_io.seek(0)

            base_name = os.path.basename(json_path).replace(".json", "")
            docx_key = f"docx/{base_name}.docx"

            s3.put_object(
                Bucket=BUCKET,
                Key=docx_key,
                Body=docx_io.getvalue(),
                ContentType=(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
            )

            logging.info("Successfully uploaded DOCX to: %s", docx_key)

            producer.send(
                "pipeline-completed",
                {
                    "document_id": document_id,
                    "final_docx_path": docx_key,
                    "source_json_path": json_path,
                    "tables_json_path": tables_json_path,
                    "status": "success",
                },
            )
            producer.flush()

        except Exception as e:
            logging.error("Error exporting DOCX: %s", e)


if __name__ == "__main__":
    main()
