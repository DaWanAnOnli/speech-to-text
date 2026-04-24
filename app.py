import uuid, asyncio, time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    WebSocket, WebSocketDisconnect, Form
)
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import RESULTS_DIR, UPLOADS_DIR, MAX_FILE_SIZE_BYTES
from transcription import load_model, transcribe_file, SUPPORTED_MODELS, _eviction_loop


# === Periodic cleanup ===

async def _cleanup_loop():
    while True:
        await asyncio.sleep(3600)
        _cleanup_old_results()


def _cleanup_old_results(max_age_seconds: float = 86400):
    now = time.time()
    for p in RESULTS_DIR.glob("*.srt"):
        try:
            if now - p.stat().st_mtime > max_age_seconds:
                p.unlink()
        except OSError:
            pass


# === Lifespan: start eviction loop at startup ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    _cleanup_old_results()
    asyncio.create_task(_eviction_loop())
    asyncio.create_task(_cleanup_loop())
    yield

app = FastAPI(title="Speech-to-Text", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


# === WebSocket connection manager ===

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.active[job_id] = ws

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

    async def send(self, job_id: str, data: dict):
        ws = self.active.get(job_id)
        if ws:
            await ws.send_json(data)

manager = ConnectionManager()


# === Allowed file extensions ===

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg", "mp4", "webm", "avi", "mov", "mkv", "wma", "aac"}


# === Routes ===

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(
        Path(__file__).parent / "templates" / "index.html",
        encoding="utf-8"
    ) as f:
        return f.read()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = Form(default="qwen"),
):
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            400,
            f"Unknown model: {model}. Supported: {', '.join(SUPPORTED_MODELS)}"
        )

    ext = Path(file.filename or "").suffix.lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    job_id = uuid.uuid4().hex
    original_ext = Path(file.filename).suffix or f".{ext}"
    temp_path = UPLOADS_DIR / f"{job_id}_orig{original_ext}"

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            400,
            f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB"
        )
    with open(temp_path, "wb") as f:
        f.write(content)

    asyncio.create_task(
        run_transcription(job_id, str(temp_path), file.filename or "unknown", model)
    )

    return {"job_id": job_id, "filename": file.filename}


async def run_transcription(job_id: str, file_path: str, filename: str, model_key: str = "qwen"):
    async def progress_callback(data: dict):
        await manager.send(job_id, data)

    try:
        await transcribe_file(file_path, filename, job_id, model_key, progress_callback)
    except Exception as exc:
        await manager.send(job_id, {
            "stage": "error",
            "message": f"Transcription failed: {exc}",
            "progress": 0
        })
    finally:
        try:
            Path(file_path).unlink()
        except OSError:
            pass


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(ws: WebSocket, job_id: str):
    await manager.connect(job_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id)


@app.get("/download/{result_id}")
async def download_result(result_id: str):
    path = RESULTS_DIR / f"{result_id}.srt"
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(
        path,
        filename=f"{result_id}.srt",
        media_type="application/x-subrip; charset=utf-8",
    )
