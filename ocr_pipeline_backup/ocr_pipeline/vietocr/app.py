import os

# PHẢI ĐẶT Ở ĐÂY - TRƯỚC KHI IMPORT BẤT KỲ THƯ VIỆN AI NÀO
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import io
import json
import logging
import time

import boto3
from PIL import Image
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BATCH_SIZE = 4


# ==========================================================
# INIT MINIO / S3
# ==========================================================

def init_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "Admin@123"),
    )


# ==========================================================
# INIT MODEL (CPU MODE)
# ==========================================================

def init_vietocr():
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["weights"] = "./weights/vgg_seq2seq.pth"
    config["cnn"]["pretrained"] = None
    config["device"] = "cpu"  # Đã chuyển sang chạy CPU

    return Predictor(config)


# ==========================================================
# DOWNLOAD & DECODE IMAGE
# ==========================================================

def get_image_from_s3(s3_client, bucket, s3_key):
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
        img_bytes = obj["Body"].read()

        if len(img_bytes) < 50:
            return None

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img

    except Exception as e:
        logging.error(f"S3 Download/Decode error for key {s3_key}: {e}")
        return None


# ==========================================================
# MAIN LOOP
# ==========================================================

def main():
    s3_client = init_s3()
    detector = init_vietocr()

    broker = os.getenv("KAFKA_BROKER", "kafka:9092")

    consumer = KafkaConsumer(
        "crop-completed",
        bootstrap_servers=[broker],
        group_id="vietocr-group",
        auto_offset_reset="latest",
        enable_auto_commit=False,
        max_poll_records=BATCH_SIZE,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks=1,
    )

    logging.info(
        f"VietOCR worker started (CPU Batching Mode - Size: {BATCH_SIZE})"
    )

    burst_start_time = None
    last_process_time = None
    total_items_processed = 0

    while True:
        msg_pack = consumer.poll(
            timeout_ms=500,
            max_records=BATCH_SIZE,
        )

        if not msg_pack:
            if (
                last_process_time is not None
                and time.time() - last_process_time >= 3.0
            ):
                total_time = last_process_time - burst_start_time

                logging.info("========== TỔNG KẾT LUỒNG XỬ LÝ ==========")
                logging.info("Đã nghỉ quá 3s, chốt luồng công việc vừa rồi:")
                logging.info(
                    f" - Tổng số ảnh đã xử lý : {total_items_processed}"
                )
                logging.info(
                    f" - Tổng thời gian chạy : {total_time:.2f}s"
                )
                logging.info(
                    "============================================"
                )

                burst_start_time = None
                last_process_time = None
                total_items_processed = 0

            continue

        if burst_start_time is None:
            burst_start_time = time.time()

        start_batch_time = time.time()

        valid_images = []
        valid_msgs = []
        invalid_msgs = []
        offsets_to_commit = {}

        s3_keys_in_batch = []

        for tp, msgs in msg_pack.items():
            for msg in msgs:
                data = msg.value

                bucket = data.get("bucket")
                s3_key = data.get("s3_key")

                s3_keys_in_batch.append(
                    s3_key if s3_key else "unknown_image"
                )

                img = None

                if bucket and s3_key:
                    img = get_image_from_s3(
                        s3_client,
                        bucket,
                        s3_key,
                    )

                if img is not None:
                    valid_images.append(img)
                    valid_msgs.append((msg, data))
                else:
                    invalid_msgs.append((msg, data))

                offsets_to_commit[tp] = OffsetAndMetadata(
                    msg.offset + 1,
                    None,
                )

        recognized_texts = []

        if valid_images:
            start_ocr = time.time()

            recognized_texts = detector.predict_batch(valid_images)

            logging.info(
                f"Inferenced {len(valid_images)} images in "
                f"{time.time() - start_ocr:.2f}s. "
                f"Images: {s3_keys_in_batch}"
            )

        for i, (msg, data) in enumerate(valid_msgs):
            payload = {
                "document_id": data.get("document_id"),
                "page": data.get("page"),
                "box_id": data.get("box_id"),
                "original_image": data.get("original_image"),
                "total_boxes": data.get("total_boxes"),
                "coordinates": data.get("coordinates"),
                "recognized_text": recognized_texts[i],
                "timestamp": time.time(),
            }

            producer.send(
                "ocr-completed",
                value=payload,
            )

        for msg, data in invalid_msgs:
            payload = {
                "document_id": data.get("document_id"),
                "page": data.get("page"),
                "original_image": data.get("original_image"),
                "box_id": data.get("box_id"),
                "total_boxes": data.get("total_boxes"),
                "coordinates": data.get("coordinates", []),
                "recognized_text": "",
                "timestamp": time.time(),
            }

            producer.send(
                "ocr-completed",
                value=payload,
            )

        producer.flush()

        if offsets_to_commit:
            consumer.commit(offsets=offsets_to_commit)

        batch_size = len(valid_msgs) + len(invalid_msgs)

        total_items_processed += batch_size
        last_process_time = time.time()

        logging.info(
            f"Batch processed {batch_size} items in "
            f"{last_process_time - start_batch_time:.2f}s"
        )


if __name__ == "__main__":
    main()