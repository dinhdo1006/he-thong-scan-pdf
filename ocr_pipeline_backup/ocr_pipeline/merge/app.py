"""Rewrite merge worker to wait for text pages + table-extract-completed."""

import os
import json
import logging
import re
import time

import boto3
from kafka import KafkaConsumer, KafkaProducer

# Prefer in-image copy; fall back to repo package when running from source.
try:
    from merge_coordinator import MergeCoordinator
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from pdf_extractor.merge_coordinator import MergeCoordinator

logging.basicConfig(level=logging.INFO)

BUCKET = os.getenv("MINIO_BUCKET", "data-lake")
REQUIRE_TABLES = os.getenv("MERGE_REQUIRE_TABLES", "1").strip() not in {"0", "false", "False"}
MERGE_TIMEOUT_S = float(os.getenv("MERGE_TIMEOUT_S", "600"))


def extract_page_number(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def init_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "Admin@123"),
    )


def emit_merge(s3, producer, coordinator: MergeCoordinator, state) -> None:
    payload = coordinator.merge_payload(state)
    document_id = payload["document_id"]
    page_paths = payload["page_json_paths"]

    merged_data = []
    for idx, key in enumerate(page_paths):
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        page_data = json.loads(obj["Body"].read().decode("utf-8"))
        page_data["page_number"] = idx + 1
        merged_data.append(page_data)

    folder = payload.get("text_folder") or ""
    folder_no_prefix = folder.replace("preprocessed/", "", 1) if folder else document_id
    output_key = f"doc_json/{folder_no_prefix}.json".replace("//", "/")

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(merged_data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

    out = {
        "document_id": document_id,
        "json_path": output_key,
        "merged_pages_count": len(merged_data),
        "tables_json_path": payload.get("tables_json_path"),
        "table_status": payload.get("table_status"),
    }
    producer.send("merge-completed", out)
    producer.flush()
    logging.info(
        "Published merge-completed: %s (tables=%s status=%s)",
        output_key,
        out.get("tables_json_path"),
        out.get("table_status"),
    )
    coordinator.pop(document_id)


def main():
    broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    s3 = init_s3()
    coordinator = MergeCoordinator(
        require_tables=REQUIRE_TABLES,
        timeout_s=MERGE_TIMEOUT_S,
    )

    consumer = KafkaConsumer(
        "document-completed",
        "table-extract-completed",
        bootstrap_servers=[broker],
        group_id="json-merge-group",
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )

    logging.info(
        "JSON Merge Worker started (require_tables=%s timeout=%ss)...",
        REQUIRE_TABLES,
        MERGE_TIMEOUT_S,
    )

    while True:
        msg_pack = consumer.poll(timeout_ms=1000, max_records=50)
        for _tp, msgs in msg_pack.items():
            for msg in msgs:
                try:
                    data = msg.value or {}
                    topic = msg.topic
                    document_id = data.get("document_id") or "unknown"
                    ready = None

                    if topic == "table-extract-completed":
                        ready = coordinator.on_tables(
                            document_id=document_id,
                            tables_json_path=data.get("tables_json_path"),
                            status=data.get("status") or "success",
                            total_pages=data.get("total_pages"),
                        )
                    else:
                        json_path = data.get("json_path")
                        if not json_path:
                            continue
                        ready = coordinator.on_page(
                            document_id=document_id,
                            json_path=json_path,
                            page_name=data.get("page_name"),
                            total_pages=data.get("total_pages"),
                            page_number=data.get("page_number"),
                        )

                    if ready is not None:
                        emit_merge(s3, producer, coordinator, ready)
                except Exception as e:
                    logging.error("Merge worker error: %s", e)

        for st in coordinator.poll_timeouts():
            try:
                emit_merge(s3, producer, coordinator, st)
            except Exception as e:
                logging.error("Merge timeout emit error: %s", e)

        # Avoid busy loop when idle
        if not msg_pack:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
