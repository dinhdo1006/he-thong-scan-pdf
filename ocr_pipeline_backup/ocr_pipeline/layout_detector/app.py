import os
import cv2
import numpy as np
import boto3
import json
import time
import logging
import requests
import base64

from paddleocr import PaddleOCR
from kafka import KafkaConsumer, KafkaProducer

logging.basicConfig(level=logging.INFO)

# =========================
# UTIL: GEOMETRY
# =========================
def normalize_box(box):
    if hasattr(box, "tolist"):
        box = box.tolist()

    if isinstance(box, list) and len(box) == 8:
        box = [
            [box[0], box[1]],
            [box[2], box[3]],
            [box[4], box[5]],
            [box[6], box[7]],
        ]
    return box

def get_center_y(box): 
    return sum(p[1] for p in box) / 4

def get_center_x(box): 
    return sum(p[0] for p in box) / 4

def get_rotate_crop_image(img, points, padding=4):
    points = np.array(points).astype(np.float32)

    w = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    ))
    h = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ))

    if w <= 0 or h <= 0:
        return None

    pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(points, pts_std)

    dst = cv2.warpPerspective(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC
    )

    if padding > 0:
        dst = cv2.copyMakeBorder(
            dst, padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    return dst

def get_intersection_table(center_x, center_y, table_boxes):
    """
    Kiểm tra xem một điểm (trọng tâm của chữ) có nằm trong bất kỳ bảng nào không.
    Trả về ID của bảng và tọa độ bảng đó nếu có.
    """
    for idx, tbox in enumerate(table_boxes):
        x1, y1, x2, y2 = tbox
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            return idx, tbox
    return None, None

# =========================
# SMART SORT (ROW AWARE - CENTER Y ALGORITHM)
# =========================
def sort_boxes(boxes):
    if not boxes:
        return []

    heights = [abs(b[3][1] - b[0][1]) for b in boxes]
    avg_height = sum(heights) / len(heights) if heights else 15
    y_thresh = avg_height * 0.5  

    boxes = sorted(boxes, key=lambda b: get_center_y(b))

    lines = []
    current = []
    prev_y = None

    for b in boxes:
        center_y = get_center_y(b)

        if prev_y is None or abs(center_y - prev_y) < y_thresh:
            current.append(b)
        else:
            lines.append(current)
            current = [b]

        prev_y = sum(get_center_y(bx) for bx in current) / len(current)

    if current:
        lines.append(current)

    result = []
    for line in lines:
        line = sorted(line, key=lambda b: get_center_x(b))
        result.extend(line)

    return result

# =========================
# MAIN
# =========================
def main():
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://minio:9000'),
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'admin'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'Admin@123')
    )

    logging.info("Loading PaddleOCR...")
    ocr = PaddleOCR(
        lang='vi',
        rec=False,            
        use_angle_cls=False,
        show_log=False
    )
    logging.info("PaddleOCR loaded")

    broker = os.getenv('KAFKA_BROKER', 'kafka:9092')

    consumer = KafkaConsumer(
        'image-arrived',
        bootstrap_servers=[broker],
        group_id='ocr-crop-group',
        auto_offset_reset='latest',
        enable_auto_commit=True,
        max_poll_records=10
    )

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        linger_ms=5
    )

    logging.info("Kafka connected. Listening for image-arrived...")

    for message in consumer:
        try:
            data = json.loads(message.value.decode('utf-8'))
            bucket = data.get('bucket', 'data-lake')
            image_path = data.get('image_path')

            if not image_path:
                continue

            logging.info(f"Processing {image_path}")

            obj = s3.get_object(Bucket=bucket, Key=image_path)
            img = cv2.imdecode(
                np.frombuffer(obj['Body'].read(), np.uint8),
                cv2.IMREAD_COLOR
            )

            if img is None:
                continue

            # 1. Table detection (optional — structured tables come from table_extract)
            table_boxes = []
            enable_roboflow = os.getenv("ENABLE_ROBOFLOW_TABLES", "0").strip() in {
                "1",
                "true",
                "True",
                "yes",
            }
            if enable_roboflow:
                _, buffer = cv2.imencode(".jpg", img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                url = os.getenv(
                    "ROBOFLOW_TABLE_URL",
                    "https://detect.roboflow.com/table-detection-u17b7/2",
                )
                api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
                if not api_key:
                    logging.warning(
                        "ENABLE_ROBOFLOW_TABLES=1 but ROBOFLOW_API_KEY is empty; skip."
                    )
                else:
                    response = requests.post(
                        url,
                        params={"api_key": api_key},
                        data=img_base64,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    )
                    api_result = response.json()
                    if "predictions" in api_result:
                        for pred in api_result["predictions"]:
                            if pred["class"] == "table" and pred["confidence"] > 0.4:
                                cx = pred["x"]
                                cy = pred["y"]
                                w = pred["width"]
                                h = pred["height"]
                                x1 = int(cx - (w / 2))
                                y1 = int(cy - (h / 2))
                                x2 = int(cx + (w / 2))
                                y2 = int(cy + (h / 2))
                                table_boxes.append([x1, y1, x2, y2])
            else:
                logging.debug(
                    "Roboflow table detect disabled (ENABLE_ROBOFLOW_TABLES=0); "
                    "use table_extract worker for structured tables."
                )

            # 2. CHẠY PADDLE ĐỂ NHẬN DIỆN CHỮ
            dt_boxes, _ = ocr.text_detector(img)

            if dt_boxes is None or len(dt_boxes) == 0:
                continue

            boxes = []
            for box in dt_boxes:
                box = normalize_box(box)
                if box and len(box) == 4:
                    boxes.append(box)

            if not boxes:
                continue

            boxes = sort_boxes(boxes)
            total_boxes = len(boxes)

            sub_path = os.path.dirname(image_path).replace("pic/", "", 1)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = f"crop/{sub_path}"

            for idx, box in enumerate(boxes):
                cropped = get_rotate_crop_image(img, box)
                if cropped is None:
                    continue

                success, buffer = cv2.imencode('.png', cropped)
                if not success:
                    continue

                img_bytes = buffer.tobytes()
                s3_key = f"{output_dir}/{base_name}_crop_{idx}.png"

                s3.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=img_bytes
                )

                # Xác định chữ này nằm ở Text thường hay trong Table
                c_x = get_center_x(box)
                c_y = get_center_y(box)
                table_idx, table_bbox = get_intersection_table(c_x, c_y, table_boxes)

                msg = {
                    "document_id": image_path.split("/")[1] if "/" in image_path else "unknown",
                    "page": base_name,
                    "original_image": image_path,
                    "bucket": bucket,
                    "s3_key": s3_key,
                    "box_id": idx,
                    "total_boxes": total_boxes,
                    "coordinates": box,
                    "layout_type": "table_cell" if table_idx is not None else "text_line",
                    "table_id": table_idx,
                    "table_bbox": table_bbox,
                    "created_at": time.time()
                }

                producer.send("crop-completed", value=msg)

            producer.flush()

        except Exception as e:
            logging.error(f"Error processing message: {e}")

if __name__ == "__main__":
    main()
