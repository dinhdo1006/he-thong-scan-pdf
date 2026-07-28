from fastapi import FastAPI, UploadFile, File, HTTPException # Thêm HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
import os, shutil, zipfile, uuid
from pathlib import Path
from preprocess import process_scan_pdf

app = FastAPI()

UPLOAD_DIR = "/data/input"
OUTPUT_DIR = "/data/output"

@app.post("/process")
async def process(file: UploadFile = File(...)):

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    safe_name = Path(file.filename).name
    unique_name = f"{uuid.uuid4()}_{safe_name}"
    pdf_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    await file.close()

    try:
        images = await run_in_threadpool(process_scan_pdf, pdf_path, OUTPUT_DIR)

        total_pages = len(images)  # 🔥 QUAN TRỌNG

        zip_path = os.path.join(OUTPUT_DIR, f"{unique_name}.zip")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for img_path in images:
                zipf.write(img_path, os.path.basename(img_path))

        # 🔥 TRẢ HEADER
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=os.path.basename(zip_path),
            headers={
                "X-Total-Pages": str(total_pages),
                "X-Document-Id": safe_name.replace(".pdf", "")
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))