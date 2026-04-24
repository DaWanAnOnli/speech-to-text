# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local web-based speech-to-text application. Users upload audio/video files through a browser, a Qwen3-ASR-1.7B model transcribes them on a GPU server, and results are downloaded as TXT files. Everything runs locally — no cloud services, no external APIs.

## Architecture

```
Windows (dev) --rsync/scp--> Ubuntu 24.04 server (GPU, e.g. RTX 5090)
                                        |
                              FastAPI backend + Vanilla HTML/JS frontend
                                        |
                              Qwen3-ASR-1.7B via qwen-asr package
                              FFmpeg for audio/video conversion
                              WebSocket for real-time progress
```

The Ubuntu server runs the FastAPI server. Windows users access it via browser at `http://<server-ip>:8000`.

## Key Files

| File | Purpose |
|---|---|
| `setup.sh` | One-time setup script for Ubuntu (run after syncing) |
| `requirements.txt` | Python pip dependencies |
| `config.py` | Environment variables (HOST, PORT, MODEL_NAME, etc.) |
| `main.py` | Uvicorn entry point (`python main.py`) |
| `app.py` | FastAPI routes, WebSocket manager |
| `transcription.py` | Model loading, FFmpeg conversion, chunked ASR pipeline |
| `templates/index.html` | Web UI page |
| `static/style.css` | UI styling (SaaS / product aesthetic) |
| `static/app.js` | UI JavaScript (drag-drop, WebSocket, download) |

## Common Commands

**Sync to Ubuntu:**
```bash
rsync -avz --exclude='venv' --exclude='__pycache__' --exclude='results' --exclude='uploads' \
  C:/Users/Lenovo/Documents/speech-to-text/ \
  user@<server-ip>:/home/user/speech-to-text/
```

**Setup on Ubuntu** (run once after syncing):
```bash
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
python main.py
```

**Start the server:**
```bash
source venv/bin/activate
python main.py
```

**Test upload via curl:**
```bash
curl -X POST -F "file=@audio.mp3" http://<server-ip>:8000/upload
```

## GPU Compatibility

- The server uses an **NVIDIA RTX 5090** (Blackwell / sm_120)
- Requires **PyTorch with CUDA 12.8** (`--index-url https://download.pytorch.org/whl/cu128`)
- If CUDA compatibility warning appears, upgrade: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

## Environment Variables (config.py)

| Variable | Default | Description |
|---|---|---|
| `STT_HOST` | `0.0.0.0` | Server bind address |
| `STT_PORT` | `8000` | Server port |
| `STT_MODEL` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model name |
| `STT_DTYPE` | `bfloat16` | Model precision |
| `STT_CHUNK_DURATION` | `30` | Audio chunk size in seconds |
| `STT_MAX_NEW_TOKENS` | `256` | Max tokens per chunk |
| `STT_MAX_FILE_MB` | `10000` | Max upload file size in MB |

## Key Implementation Notes

- **Singleton model**: loaded once at server startup, shared across all requests
- **Chunked transcription**: audio split into 30s chunks, transcribed sequentially, progress reported per chunk
- **Progress via WebSocket**: each upload gets a job_id; client connects to `/ws/{job_id}` for live stage updates
- **Download**: uses File System Access API (`showSaveFilePicker`) in Chrome/Edge for OS save dialog; falls back to Blob download in other browsers
- **Server cleanup**: results deleted from server after download. Periodic cleanup every 1 hour + on startup removes files older than 24h
- **No history**: no persistence between sessions — files are cleaned up after each transcription

## Development Notes

- Frontend is pure vanilla HTML/CSS/JS — no build step, no frameworks
- Uploaded files are deleted after transcription starts (`finally` block in `run_transcription`)
- FFmpeg must be installed on the server (`apt install ffmpeg`)
- Model is cached at `~/.cache/huggingface/` after first download
