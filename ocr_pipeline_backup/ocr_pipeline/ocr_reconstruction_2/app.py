import os
import json
import time
import logging

import boto3
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata

logging.basicConfig(level=logging.INFO)

TIMEOUT = int(os.getenv("RECONSTRUCT_TIMEOUT", 120))
BUCKET = os.getenv("MINIO_BUCKET", "data-lake")

# Buffer lưu trữ trên RAM
document_buffer = {}

# =========================
# GEOMETRY UTILS
# =========================
def get_center_y(box): return sum(p[1] for p in box) / 4
def get_center_x(box): return sum(p[0] for p in box) / 4
def get_height(box):
    y_coords = [p[1] for p in box]
    return max(y_coords) - min(y_coords)

def sort_boxes(boxes):
    valid_boxes = [b for b in boxes if b.get("coordinates") and len(b["coordinates"]) == 4]
    
    return sorted(valid_boxes, key=lambda x: (
        get_center_y(x['coordinates']),
        get_center_x(x['coordinates'])
    ))

def group_lines(boxes):
    if not boxes:
        return []

    avg_height = sum(get_height(b['coordinates']) for b in boxes) / len(boxes) if boxes else 15
    y_threshold = avg_height * 0.5

    lines, current = [], []
    prev_y = None

    for b in boxes:
        y = get_center_y(b['coordinates'])

        if prev_y is None or abs(y - prev_y) < y_threshold:
            current.append(b)
        else:
            lines.append(current)
            current = [b]
        
        prev_y = sum(get_center_y(box['coordinates']) for box in current) / len(current)

    if current:
        lines.append(current)

    for line in lines:
        line.sort(key=lambda x: get_center_x(x['coordinates']))

    return lines

# =========================
# OUTPUT BUILDERS (HỖ TRỢ TABLE)
# =========================
def build_text(lines, table_groups):
    """
    Ghép nội dung thành văn bản thuần túy. 
    Các vùng Table sẽ được hiển thị phân tách rõ ràng để dễ đọc.
    """
    if not lines and not table_groups:
        return ""

    all_boxes = [b for line in lines for b in line if b.get("coordinates")]
    if not all_boxes:
        return ""

    min_x_global = min(
        min(p[0] for p in b["coordinates"])
        for b in all_boxes
    )

    CHAR_WIDTH = 45  
    result = ""

    for line in lines:
        line_str = ""
        for i, b in enumerate(line):
            text = b.get("recognized_text", "")
            if not text:
                continue

            coords = b["coordinates"]
            x_min = min(p[0] for p in coords)

            if i == 0:
                indent_pixels = x_min - min_x_global
                indent_spaces = int(indent_pixels / CHAR_WIDTH)
                line_str += " " * max(0, indent_spaces)
            else:
                prev = line[i - 1]
                prev_coords = prev["coordinates"]
                prev_x_max = max(p[0] for p in prev_coords)
                gap_pixels = x_min - prev_x_max
                gap_spaces = int(gap_pixels / CHAR_WIDTH)
                line_str += " " * max(1, gap_spaces)

            line_str += text

        result += line_str.rstrip() + "\n"

    # Nếu có các bảng được nhận diện, append phần cấu trúc bảng xuống dưới hoặc lồng ghép
    if table_groups:
        result += "\n--- BẢNG BIỂU PHÁT HIỆN ---\n"
        for t_id, t_boxes in table_groups.items():
            result += f"[Bảng #{t_id}]\n"
            # Sắp xếp các ô trong bảng theo dòng và cột
            t_lines = group_lines(t_boxes)
            for t_line in t_lines:
                row_text = " | ".join(tb.get("recognized_text", "") for tb in t_line)
                result += f"  | {row_text} |\n"

    return result

def build_json(image, lines, table_groups):
    """
    Đóng gói JSON cấu trúc phân tách rõ ràng giữa Text thông thường và Table Cells.
    """
    formatted_tables = []
    for t_id, t_boxes in table_groups.items():
        t_lines = group_lines(t_boxes)
        formatted_tables.append({
            "table_id": t_id,
            "lines": [
                {
                    "line_id": i,
                    "cells": [
                        {
                            "box_id": b.get("box_id"),
                            "text": b.get("recognized_text", ""),
                            "coordinates": b.get("coordinates")
                        } for b in t_line
                    ]
                } for i, t_line in enumerate(t_lines)
            ]
        })

    return {
        "source_image": image,
        "processed_at": time.time(),
        "lines": [
            {
                "line_id": i,
                "line_text": " ".join(b.get("recognized_text", "") for b in line if b.get("recognized_text")).strip(),
                "words": [
                    {
                        "box_id": b.get("box_id"),
                        "text": b.get("recognized_text", ""),
                        "coordinates": b.get("coordinates")
                    } for b in line
                ]
            }
            for i, line in enumerate(lines)
            # LƯU Ý KỸ: Đã xóa đoạn điều kiện if b.get(...) ở đây
        ],
        "tables": formatted_tables
    }

# =========================
# MAIN LOOP
# =========================
def main():
    broker = os.getenv("KAFKA_BROKER", "kafka:9092")

    consumer = KafkaConsumer(
        "ocr-completed",
        bootstrap_servers=[broker],
        group_id="reconstruct-group",
        auto_offset_reset="latest",
        enable_auto_commit=False,
        max_poll_records=100,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "Admin@123")
    )

    logging.info("Reconstruct worker started (with Table Support)...")

    while True:
        msg_pack = consumer.poll(timeout_ms=500)
        offsets = {}

        for tp, msgs in msg_pack.items():
            for msg in msgs:
                try:
                    data = msg.value
                    image = data.get("original_image")
                    box_id = data.get("box_id")
                    total = data.get("total_boxes")

                    if image is None or box_id is None:
                        continue

                    if image not in document_buffer:
                        document_buffer[image] = {
                            "first_seen": time.time(),
                            "boxes": {},
                            "total_boxes": total,
                            "document_id": data.get("document_id", "unknown")
                        }

                    buf = document_buffer[image]
                    
                    if buf["total_boxes"] is None and total is not None:
                        buf["total_boxes"] = total

                    buf["boxes"][box_id] = data

                    offsets[tp] = OffsetAndMetadata(msg.offset + 1, None)

                except Exception as e:
                    logging.error(f"Error parsing message: {e}")

        now = time.time()
        
        for image, buf in list(document_buffer.items()):
            received = len(buf["boxes"])
            total = buf["total_boxes"]

            is_timeout = (now - buf["first_seen"]) > TIMEOUT
            is_done = (total is not None) and (received >= total)

            if is_done or is_timeout:
                if is_timeout:
                    logging.warning(f"TIMEOUT {image}: Gathered {received}/{total} boxes.")
                else:
                    logging.info(f"SUCCESS {image}: Gathered {received}/{total} boxes.")

                try:
                    all_boxes = list(buf["boxes"].values())
                    
                    # Phân loại box nào thuộc Text thường, box nào nằm trong Bảng
                    normal_boxes = []
                    table_groups = {} # Key: table_id, Value: list of boxes
                    
                    for b in all_boxes:
                        layout_type = b.get("layout_type", "text_line")
                        if layout_type == "table_cell":
                            t_id = b.get("table_id", 0)
                            if t_id not in table_groups:
                                table_groups[t_id] = []
                            table_groups[t_id].append(b)
                        else:
                            normal_boxes.append(b)

                    # Xử lý hình học cho text thường
                    sorted_normal = sort_boxes(normal_boxes)
                    lines = group_lines(sorted_normal)

                    text_content = build_text(lines, table_groups)
                    json_content = build_json(image, lines, table_groups)

                    base_name = os.path.splitext(os.path.basename(image))[0]
                    sub_path = os.path.dirname(image).replace("pic/", "", 1)
                    
                    txt_key = f"text/{sub_path}/{base_name}.txt"
                    json_key = f"preprocessed/{sub_path}/{base_name}.json"

                    s3.put_object(Bucket=BUCKET, Key=txt_key, Body=text_content.encode("utf-8"))
                    s3.put_object(Bucket=BUCKET, Key=json_key, Body=json.dumps(json_content, ensure_ascii=False).encode("utf-8"))

                    producer.send(
                        "document-completed",
                        key=None,
                        value={
                            "document_id": buf["document_id"],
                            "page_name": base_name,
                            "original_image": image,
                            "text_path": txt_key,
                            "json_path": json_key,
                            "status": "partial" if is_timeout else "success",
                            "boxes_processed": received,
                            "total_boxes": total
                        }
                    )

                except Exception as e:
                    logging.error(f"Error saving/sending reconstruct for {image}: {e}")

                finally:
                    del document_buffer[image]

        if offsets:
            consumer.commit(offsets=offsets)
            producer.flush()

if __name__ == "__main__":
    main()