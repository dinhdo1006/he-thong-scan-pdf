import os
from pdf2image import convert_from_path
import cv2
import numpy as np

def process_scan_pdf(pdf_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=300)

    paths = []

    for i, page in enumerate(pages):

        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        out = os.path.join(output_dir, f"page_{i+1}.png")

        cv2.imwrite(out, gray)

        paths.append(out)

    return paths