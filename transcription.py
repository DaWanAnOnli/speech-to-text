import os, subprocess, asyncio, json, uuid
from pathlib import Path
from typing import Optional, Callable, Dict, List
from datetime import datetime

import torch

from config import RESULTS_DIR, UPLOADS_DIR, MODEL_NAME, DTYPE, CHUNK_DURATION_SEC, FORCE_ALIGNER_NAME

# === Singleton model manager ===

_model_instance = None
_model_lock = asyncio.Lock()


async def load_model(
    progress_callback: Optional[Callable] = None
):
    global _model_instance
    async with _model_lock:
        if _model_instance is not None:
            return _model_instance

        if progress_callback:
            await progress_callback({
                "stage": "loading_model",
                "message": "Loading Qwen3-ASR model...",
                "progress": 0
            })

        from qwen_asr import Qwen3ASRModel
        from config import DTYPE, MAX_NEW_TOKENS

        if progress_callback:
            await progress_callback({
                "stage": "loading_model",
                "message": "Initializing model...",
                "progress": 50
            })

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, DTYPE, torch.bfloat16) if device == "cuda" else torch.float32

        _model_instance = Qwen3ASRModel.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            device_map=device,
            max_new_tokens=MAX_NEW_TOKENS,
            forced_aligner=FORCE_ALIGNER_NAME,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device),
        )

        if progress_callback:
            await progress_callback({
                "stage": "loading_model",
                "message": "Model ready",
                "progress": 100
            })

        return _model_instance


def get_model():
    return _model_instance


# === FFmpeg helpers ===

def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _merge_segments(segments: List[dict], gap_threshold: float = 0.5, max_duration: float = 7.0) -> List[dict]:
    if not segments:
        return []
    merged: List[dict] = []
    current = dict(segments[0])
    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        text = (current["text"] + " " + seg["text"]).strip()
        duration = seg["end"] - current["start"]
        if gap <= gap_threshold and duration <= max_duration:
            current = {"start": current["start"], "end": seg["end"], "text": text}
        else:
            merged.append(current)
            current = dict(seg)
    merged.append(current)
    return merged


def segments_to_srt(segments: List[dict]) -> str:
    merged = _merge_segments(segments)
    lines = []
    for i, seg in enumerate(merged, 1):
        text = seg["text"].strip()
        if not text:
            continue
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(lines) + "\n"


def _run_ffmpeg(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    return result


def convert_to_wav(input_path: str, output_path: str) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ])


def get_audio_duration(wav_path: str) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


# === Core transcription pipeline ===

async def transcribe_file(
    file_path: str,
    original_filename: str,
    job_id: str,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    temp_wav = UPLOADS_DIR / f"{job_id}_converted.wav"

    # ---- Stage 1: FFmpeg Conversion ----
    if progress_callback:
        await progress_callback({
            "stage": "converting",
            "message": "Converting to WAV...",
            "progress": 5
        })

    convert_to_wav(file_path, str(temp_wav))

    if progress_callback:
        await progress_callback({
            "stage": "converting",
            "message": "Conversion complete",
            "progress": 15
        })

    # ---- Stage 2: Load Model ----
    model = await load_model(progress_callback)

    # ---- Stage 3: Chunked Transcription ----
    if progress_callback:
        await progress_callback({
            "stage": "transcribing",
            "message": "Starting transcription...",
            "progress": 20
        })

    duration = get_audio_duration(str(temp_wav))
    chunk_sec = CHUNK_DURATION_SEC
    total_chunks = max(1, int(duration / chunk_sec) + (1 if duration % chunk_sec > 0 else 0))

    all_texts: List[str] = []
    all_segments: List[dict] = []
    detected_language: Optional[str] = None

    for chunk_idx in range(total_chunks):
        start_sec = chunk_idx * chunk_sec
        end_sec = min((chunk_idx + 1) * chunk_sec, duration)

        chunk_text, chunk_lang, chunk_segments = await _transcribe_chunk(
            model, str(temp_wav), start_sec, end_sec
        )

        if chunk_text.strip():
            all_texts.append(chunk_text)
        all_segments.extend(chunk_segments)
        if detected_language is None and chunk_lang:
            detected_language = chunk_lang

        chunk_progress = 20 + int((chunk_idx + 1) / total_chunks * 60)
        if progress_callback:
            await progress_callback({
                "stage": "transcribing",
                "message": f"Chunk {chunk_idx + 1}/{total_chunks} ({end_sec:.0f}s / {duration:.0f}s)...",
                "progress": chunk_progress,
                "chunk": chunk_idx + 1,
                "total_chunks": total_chunks,
            })

    full_text = " ".join(all_texts)

    # ---- Stage 4: Save Results ----
    if progress_callback:
        await progress_callback({
            "stage": "saving",
            "message": "Saving results...",
            "progress": 85
        })

    timestamp = datetime.now().isoformat()
    safe_name = Path(original_filename).stem.replace(" ", "_")
    result_id = f"{safe_name}_{job_id[:8]}"

    srt_path = RESULTS_DIR / f"{result_id}.srt"
    srt_path.write_text(segments_to_srt(all_segments), encoding="utf-8")

    json_data = {
        "id": result_id,
        "original_filename": original_filename,
        "transcription": full_text,
        "detected_language": detected_language or "unknown",
        "timestamp": timestamp,
        "duration_seconds": round(duration, 2),
        "chunks_processed": total_chunks,
        "segments": all_segments,
    }
    json_path = RESULTS_DIR / f"{result_id}.json"
    json_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if progress_callback:
        await progress_callback({
            "stage": "saving",
            "message": "Results saved",
            "progress": 95
        })

    # ---- Cleanup ----
    try:
        temp_wav.unlink()
    except OSError:
        pass

    if progress_callback:
        await progress_callback({
            "stage": "complete",
            "message": "Transcription complete",
            "progress": 100,
            "result_id": result_id,
            "text": full_text,
            "language": detected_language,
        })

    return {
        "id": result_id,
        "text": full_text,
        "language": detected_language,
        "timestamp": timestamp,
        "srt_path": str(srt_path),
        "json_path": str(json_path),
    }


async def _transcribe_chunk(
    model,
    wav_path: str,
    start_sec: float,
    end_sec: float,
) -> tuple[str, Optional[str], List[dict]]:
    def _run() -> tuple[str, Optional[str], List[dict]]:
        chunk_id = uuid.uuid4().hex
        chunk_file = UPLOADS_DIR / f"chunk_{chunk_id}.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(start_sec), "-to", str(end_sec),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(chunk_file)
            ], capture_output=True, text=True)

            results = model.transcribe(str(chunk_file), return_time_stamps=True)

            first = results[0] if results else None
            text = getattr(first, "text", "") or "" if first else ""
            lang = getattr(first, "language", None) or getattr(first, "lang", None)

            segments: List[dict] = []
            time_stamps = getattr(first, "time_stamps", []) or []
            for ts in time_stamps:
                seg_text = getattr(ts, "text", "") or ""
                seg_start = getattr(ts, "start_time", None)
                seg_end = getattr(ts, "end_time", None)
                if seg_start is not None and seg_end is not None:
                    segments.append({
                        "text": seg_text,
                        "start": start_sec + float(seg_start),
                        "end": start_sec + float(seg_end),
                    })

            return str(text), lang, segments
        finally:
            try:
                chunk_file.unlink()
            except OSError:
                pass

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)
