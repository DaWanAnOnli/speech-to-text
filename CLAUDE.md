# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DO THIS FOR EVERY INITIAL REQUEST: Analyze my request. Use whatever strategies you think will produce the best results. You have 1 million context to play with, so use it optimally. Don't forget to leverage web search, multi-agents, read files, etc.
Don't go straight to implementation. Produce a comprehensive plan, ask clarifying questions where needed. Be absolutely honest with your evaluations, do not glaze the user.

DO THIS AFTER EVERY CHANGE IN CODEBASE:
- Update this file to reflect the changes
- Update setup.sh to reflect the changes. setup.sh must be able to be run for either initial setup from scratch, or when there are new features/updates to the codebase.


## Project Overview

A local web-based speech-to-text application. Users upload audio/video files through a browser, a transcription model (Qwen3-ASR-1.7B or Whisper-large-v3) transcribes them on a GPU server, and results are downloaded as SRT files. Everything runs locally — no cloud services, no external APIs.

## Architecture

```
Windows (dev) --rsync/scp--> Ubuntu 24.04 server (GPU, e.g. RTX 5090)
                                        |
                              FastAPI backend + Vanilla HTML/JS frontend
                                        |
                              Transcription model (Qwen3-ASR-1.7B or Whisper-large-v3)
                              FFmpeg for audio/video conversion
                              WebSocket for real-time progress
```

The Ubuntu server runs the FastAPI server. Windows users access it via browser at `http://<server-ip>:8000`.

## Supported Models

| Key | Display Name | HuggingFace ID | Notes |
|---|---|---|---|
| `qwen` | Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B` | Default model |
| `whisper` | Whisper-large-v3 | `openai/whisper-large-v3` | Via whisper-timestamped |

## Key Files

| File | Purpose |
|---|---|
| `setup.sh` | One-time setup script for Ubuntu (run after syncing) |
| `requirements.txt` | Python pip dependencies |
| `config.py` | Environment variables (HOST, PORT, CHUNK_DURATION, etc.) |
| `main.py` | Uvicorn entry point (`python main.py`) |
| `app.py` | FastAPI routes, WebSocket manager |
| `transcription.py` | Model loading (pool), FFmpeg conversion, chunked ASR pipeline |
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
curl -X POST -F "file=@audio.mp3" -F "model=qwen" http://<server-ip>:8000/upload
```

## GPU Compatibility

- The server uses an **NVIDIA RTX 5090** (Blackwell / sm_120)
- Requires **PyTorch with CUDA 12.8** (`--index-url https://download.pytorch.org/whl/cu128`)
- If CUDA compatibility warning appears, upgrade: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- **Whisper-large-v3** uses `whisper_timestamped` with default attention (SDPA-compatible) — no Flash Attention required

## Environment Variables (config.py)

| Variable | Default | Description |
|---|---|---|
| `STT_HOST` | `0.0.0.0` | Server bind address |
| `STT_PORT` | `8000` | Server port |
| `STT_DTYPE` | `bfloat16` | Model precision (Qwen only) |
| `STT_CHUNK_DURATION` | `30` | Audio chunk size in seconds |
| `STT_MAX_NEW_TOKENS` | `256` | Max tokens per chunk (Qwen only) |
| `STT_MAX_FILE_MB` | `10000` | Max upload file size in MB |

## Key Implementation Notes

- **Model pool with idle eviction**: models load on first use, stay cached for 30 minutes of idle, then are evicted. Check runs every 5 minutes.
- **Model selection at upload time**: the frontend sends a `model` field with the upload. Both `qwen` and `whisper` are available.
- **Filename includes model key**: result files are named `{original_stem}_{job_id}_{model_key}.srt` (e.g., `myfile_a1b2c3d4_qwen.srt`).
- **SRT output only**: only `.srt` files are generated — no JSON sidecar.
- **Chunked transcription**: audio split into 30s chunks, transcribed sequentially, progress reported per chunk via WebSocket.
- **Progress via WebSocket**: each upload gets a `job_id`; client connects to `/ws/{job_id}` for live stage updates.
- **No history**: no persistence between sessions — SRT files are cleaned up after 24 hours.

## WebSocket Protocol

Server sends JSON messages with these shapes:

| Stage | Progress | Keys |
|---|---|---|
| `loading_model` | 0→100 | `stage`, `message`, `progress` |
| `converting` | 5→15 | `stage`, `message`, `progress` |
| `transcribing` | 20→80 | `stage`, `message`, `progress`, `chunk`, `total_chunks` |
| `saving` | 85→95 | `stage`, `message`, `progress` |
| `complete` | 100 | `stage`, `message`, `progress`, `result_id`, `filename`, `text`, `language`, `model_key`, `model_name` |
| `error` | 0 | `stage`, `message`, `progress` |

## Development Notes

- Frontend is pure vanilla HTML/CSS/JS — no build step, no frameworks
- Uploaded files are deleted after transcription starts (`finally` block in `run_transcription`)
- FFmpeg must be installed on the server (`apt install ffmpeg`)
- Models are cached at `~/.cache/huggingface/` after first download
- Qwen model + ForcedAligner cache: ~4.6GB; Whisper-large-v3 cache: ~3GB
