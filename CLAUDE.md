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
                              FFmpeg: any format -> 16kHz mono WAV
                                        |
                              [Optional] Sepformer: Noise Reduction
                                        |
                              [Optional] MetricGAN+: Audio Enhancement
                                        |
                              Transcription model (Qwen3-ASR-1.7B or Whisper-large-v3)
                                        |
                              WebSocket for real-time progress
```

Enhancement stages are optional and independently controllable per-request.

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
curl -X POST \
  -F "file=@audio.mp3" \
  -F "model=qwen" \
  -F "enable_noise_reduction=true" \
  -F "enable_audio_enhancement=true" \
  http://<server-ip>:8000/upload
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
| `STT_ENABLE_NOISE_REDUCTION` | `true` | Enable MP-SENet noise reduction by default |
| `STT_ENABLE_AUDIO_ENHANCEMENT` | `true` | Enable MetricGAN+ enhancement by default |
| `STT_NOISE_REDUCTION_MODEL` | `speechbrain/sepformer-wham16k-enhancement` | HuggingFace model for noise reduction |
| `STT_ENHANCEMENT_MODEL` | `speechbrain/metricgan-plus-voicebank` | HuggingFace model for audio enhancement |
| `STT_MAX_CONCURRENT_JOBS` | `2` | Max simultaneous transcription jobs on GPU (to prevent OOM) |

## Key Implementation Notes

- **Model pool with idle eviction**: transcription models load on first use, stay cached for 30 minutes of idle, then are evicted. Enhancement models are singletons loaded once and kept in memory.
- **Model selection at upload time**: the frontend sends a `model` field with the upload. Both `qwen` and `whisper` are available.
- **Per-request enhancement toggles**: the frontend sends `enable_noise_reduction` and `enable_audio_enhancement` fields with each upload (both default to `true`). The server falls back to env var defaults if not provided.
- **Enhancement pipeline**: FFmpeg converts to 16kHz mono WAV, then optionally applies Sepformer noise reduction and/or MetricGAN+ enhancement, then transcribes. Applied to the full file once before chunking.
- **Filename includes model key**: result files are named `{original_stem}_{job_id}_{model_key}.srt` (e.g., `myfile_a1b2c3d4_qwen.srt`).
- **SRT output only**: only `.srt` files are generated — no JSON sidecar.
- **Chunked transcription**: audio split into 30s chunks, transcribed sequentially, progress reported per chunk via WebSocket.
- **Progress via WebSocket**: each upload gets a `job_id`; client connects to `/ws/{job_id}` for live stage updates.
- **No history**: no persistence between sessions — SRT files are cleaned up after 24 hours.
- **Enhancement is non-fatal**: if enhancement fails, the pipeline logs a warning and continues with the original (non-enhanced) audio.
- **Parallel multi-file processing**: multiple files can be uploaded simultaneously (file input has `multiple` attribute, drag-drop accepts multiple files). Each upload gets its own job card with independent progress. The server enforces a concurrency limit of `MAX_CONCURRENT_JOBS` (default 2) via a semaphore — excess uploads queue indefinitely and receive `queued` stage messages until a slot frees up. Each job card has a remove (X) button and a download button (on completion).
- **GPU memory guard**: `_check_gpu_memory()` is called at the start of each transcription. If GPU memory is nearly exhausted, it raises `RuntimeError`, which is caught by `run_transcription` and sent as an `error` stage to the frontend. This prevents OOM crashes on long files.

## WebSocket Protocol

Server sends JSON messages with these shapes:

| Stage | Progress | Keys |
|---|---|---|
| `queued` | 0 | `stage`, `message`, `progress` — sent when job is accepted but no GPU slot available |
| `queued_starting` | 0 | `stage`, `message`, `progress` — sent when GPU slot acquired and transcription begins |
| `loading_model` | 0→100 | `stage`, `message`, `progress` |
| `converting` | 5→15 | `stage`, `message`, `progress` |
| `noise_reducing` | 16→35 | `stage`, `message`, `progress` (only if enabled) |
| `enhancing` | 16→35 | `stage`, `message`, `progress` (only if enabled) |
| `transcribing` | 50→92 | `stage`, `message`, `progress`, `chunk`, `total_chunks` |
| `saving` | 92→98 | `stage`, `message`, `progress` |
| `complete` | 100 | `stage`, `message`, `progress`, `result_id`, `filename`, `text`, `language`, `model_key`, `model_name` |
| `error` | 0 | `stage`, `message`, `progress` |

## Server Logging

The server uses a structured logger (`speech-to-text`) that outputs:
- Job start with model name, active enhancements, and filename
- Success logs when noise reduction and/or audio enhancement complete
- Warnings when enhancement fails (pipeline continues with raw audio)

Example output:
```
speech-to-text | [5d7dc88a] Starting transcription | model=Qwen3-ASR-1.7B | enhancements=noise_reduction, audio_enhancement | file=video.mp4
speech-to-text | [5d7dc88a] Noise reduction applied
speech-to-text | [5d7dc88a] Audio enhancement applied
```

## Development Notes

- Frontend is pure vanilla HTML/CSS/JS — no build step, no frameworks
- Uploaded files are deleted after transcription starts (`finally` block in `run_transcription`)
- FFmpeg must be installed on the server (`apt install ffmpeg`)
- Models are cached at `~/.cache/huggingface/` after first download; subsequent runs are fully offline
- Qwen model + ForcedAligner cache: ~4.6GB; Whisper-large-v3 cache: ~3GB; Sepformer noise reduction cache: ~300MB; MetricGAN+ cache: ~300MB
- Enhancement models (Sepformer, MetricGAN+) use `speechbrain.inference` API (not the deprecated `speechbrain.pretrained`)
