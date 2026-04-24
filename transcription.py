import os, subprocess, asyncio, json, uuid, time
from pathlib import Path
from typing import Optional, Callable, Dict, List
from datetime import datetime

import torch
import numpy as np
import soundfile as sf

from config import (RESULTS_DIR, UPLOADS_DIR, MODEL_NAME, DTYPE, CHUNK_DURATION_SEC,
    FORCE_ALIGNER_NAME, ENABLE_NOISE_REDUCTION, ENABLE_AUDIO_ENHANCEMENT,
    NOISE_REDUCTION_MODEL, ENHANCEMENT_MODEL)

# === Supported models ===

SUPPORTED_MODELS = {
    "qwen": {
        "display_name": "Qwen3-ASR-1.7B",
        "hf_name": "Qwen/Qwen3-ASR-1.7B",
    },
    "whisper": {
        "display_name": "Whisper-large-v3",
        "hf_name": "openai/whisper-large-v3",
    },
}

# === Model pool with idle eviction ===

_loaded_models: dict[str, object] = {}
_model_last_used: dict[str, float] = {}
_model_lock = asyncio.Lock()
_IDLE_TIMEOUT_SEC = 30 * 60


async def load_model(
    model_key: str,
    progress_callback: Optional[Callable] = None,
):
    global _loaded_models, _model_last_used

    async with _model_lock:
        if model_key in _loaded_models:
            _model_last_used[model_key] = time.time()
            return _loaded_models[model_key]

        cfg = SUPPORTED_MODELS[model_key]

        if progress_callback:
            await progress_callback({
                "stage": "loading_model",
                "message": f"Loading {cfg['display_name']}...",
                "progress": 0,
            })

        if model_key == "qwen":
            model = await _load_qwen_model(cfg, progress_callback)
        elif model_key == "whisper":
            model = await _load_whisper_model(cfg, progress_callback)
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        _loaded_models[model_key] = model
        _model_last_used[model_key] = time.time()
        return model


def get_model(model_key: str):
    return _loaded_models.get(model_key)


# === Idle eviction loop ===

async def _eviction_loop():
    while True:
        await asyncio.sleep(300)
        now = time.time()
        async with _model_lock:
            for key in list(_loaded_models.keys()):
                if now - _model_last_used.get(key, 0) > _IDLE_TIMEOUT_SEC:
                    del _loaded_models[key]
                    del _model_last_used[key]


# === Enhancement singletons (shared across all transcription models) ===

_noise_reduction_model = None
_noise_reduction_lock = asyncio.Lock()
_enhancement_model = None
_enhancement_lock = asyncio.Lock()


async def load_noise_reduction_model(progress_callback=None):
    global _noise_reduction_model
    async with _noise_reduction_lock:
        if _noise_reduction_model is not None:
            return _noise_reduction_model
        if progress_callback:
            await progress_callback({
                "stage": "noise_reducing",
                "message": "Loading noise reduction model...",
                "progress": 16
            })
        from speechbrain.inference.separation import SepformerSeparation
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        _noise_reduction_model = SepformerSeparation.from_hparams(
            source=NOISE_REDUCTION_MODEL,
            run_opts={"device": device},
        )
        return _noise_reduction_model


async def load_enhancement_model(progress_callback=None):
    global _enhancement_model
    async with _enhancement_lock:
        if _enhancement_model is not None:
            return _enhancement_model
        if progress_callback:
            await progress_callback({
                "stage": "enhancing",
                "message": "Loading enhancement model...",
                "progress": 26
            })
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        _enhancement_model = SpectralMaskEnhancement.from_hparams(
            source=ENHANCEMENT_MODEL,
            run_opts={"device": device},
        )
        return _enhancement_model


# === Model loaders ===

async def _load_qwen_model(
    cfg: dict,
    progress_callback: Optional[Callable] = None,
):
    from qwen_asr import Qwen3ASRModel
    from config import DTYPE, MAX_NEW_TOKENS

    if progress_callback:
        await progress_callback({
            "stage": "loading_model",
            "message": "Initializing model...",
            "progress": 50,
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, DTYPE, torch.bfloat16) if device == "cuda" else torch.float32

    model = Qwen3ASRModel.from_pretrained(
        cfg["hf_name"],
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
            "progress": 100,
        })

    return model


async def _load_whisper_model(
    cfg: dict,
    progress_callback: Optional[Callable] = None,
):
    import whisper_timestamped as whisper

    if progress_callback:
        await progress_callback({
            "stage": "loading_model",
            "message": "Initializing Whisper...",
            "progress": 50,
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large-v3", device=device)

    if progress_callback:
        await progress_callback({
            "stage": "loading_model",
            "message": "Model ready",
            "progress": 100,
        })

    return model


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


def _normalize_audio(audio: "np.ndarray") -> "np.ndarray":
    """Normalize audio to prevent clipping."""
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        return audio / max_val
    return audio


def apply_noise_reduction(input_wav_path: str, output_wav_path: str, noise_reducer) -> None:
    """Apply Sepformer noise reduction to a WAV file."""
    est_sources = noise_reducer.separate_file(path=input_wav_path)
    # est_sources shape: (batch, time) or (batch, time, channels)
    if est_sources.dim() == 3:
        audio_tensor = est_sources[:, :, 0].squeeze(0)
    else:
        audio_tensor = est_sources.squeeze(0)
    output_audio = audio_tensor.cpu().numpy()

    output_audio = _normalize_audio(output_audio)
    sf.write(output_wav_path, output_audio, 16000, subtype='PCM_16')


def apply_enhancement(input_wav_path: str, output_wav_path: str, enhancer) -> None:
    """Apply MetricGAN+ enhancement to a WAV file."""
    enhanced_tensor = enhancer.enhance_batch(
        enhancer.load_audio(input_wav_path).unsqueeze(0),
        lengths=torch.tensor([1.0])
    )
    # enhanced_tensor shape: (batch, time) or (batch, time, channels)
    if enhanced_tensor.dim() == 3:
        enhanced_audio = enhanced_tensor[:, :, 0].squeeze(0)
    else:
        enhanced_audio = enhanced_tensor.squeeze(0)
    enhanced_audio = enhanced_audio.cpu().numpy()

    enhanced_audio = _normalize_audio(enhanced_audio)
    sf.write(output_wav_path, enhanced_audio, 16000, subtype='PCM_16')


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
    model_key: str = "qwen",
    enable_noise_reduction: bool = None,
    enable_audio_enhancement: bool = None,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    # Resolve None defaults to config values
    if enable_noise_reduction is None:
        enable_noise_reduction = ENABLE_NOISE_REDUCTION
    if enable_audio_enhancement is None:
        enable_audio_enhancement = ENABLE_AUDIO_ENHANCEMENT
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

    # ---- Stage 2: Enhancement ----
    if enable_noise_reduction or enable_audio_enhancement:
        enhanced_wav = UPLOADS_DIR / f"{job_id}_enhanced.wav"

        if enable_noise_reduction and enable_audio_enhancement:
            # Both enabled: noise reduction first, then enhancement
            temp_denoised = UPLOADS_DIR / f"{job_id}_denoised.wav"
            try:
                if progress_callback:
                    await progress_callback({
                        "stage": "noise_reducing",
                        "message": "Loading noise reduction model...",
                        "progress": 16
                    })
                noise_reducer = await load_noise_reduction_model()
                if progress_callback:
                    await progress_callback({
                        "stage": "noise_reducing",
                        "message": "Applying noise reduction...",
                        "progress": 18
                    })
                apply_noise_reduction(str(temp_wav), str(temp_denoised), noise_reducer)

                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": "Loading enhancement model...",
                        "progress": 20
                    })
                enhancer_model = await load_enhancement_model()
                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": "Applying audio enhancement...",
                        "progress": 22
                    })
                apply_enhancement(str(temp_denoised), str(enhanced_wav), enhancer_model)

                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": "Enhancement complete",
                        "progress": 35
                    })
                temp_denoised.unlink(missing_ok=True)
                temp_wav.unlink(missing_ok=True)
                temp_wav = enhanced_wav
            except Exception as exc:
                import logging
                logging.warning(f"Enhancement failed, using raw audio: {exc}")
                for f in [enhanced_wav, temp_denoised]:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": f"Enhancement skipped: {exc}",
                        "progress": 35
                    })

        elif enable_noise_reduction:
            denoised_wav = UPLOADS_DIR / f"{job_id}_denoised.wav"
            try:
                if progress_callback:
                    await progress_callback({
                        "stage": "noise_reducing",
                        "message": "Loading noise reduction model...",
                        "progress": 16
                    })
                noise_reducer = await load_noise_reduction_model()
                if progress_callback:
                    await progress_callback({
                        "stage": "noise_reducing",
                        "message": "Applying noise reduction...",
                        "progress": 18
                    })
                apply_noise_reduction(str(temp_wav), str(denoised_wav), noise_reducer)
                if progress_callback:
                    await progress_callback({
                        "stage": "noise_reducing",
                        "message": "Noise reduction complete",
                        "progress": 35
                    })
                temp_wav.unlink()
                temp_wav = denoised_wav
            except Exception as exc:
                import logging
                logging.warning(f"Noise reduction failed, using raw audio: {exc}")
                try:
                    denoised_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                if progress_callback:
                    await progress_callback({
                        "stage": "noise_reducing",
                        "message": f"Noise reduction skipped: {exc}",
                        "progress": 35
                    })

        elif enable_audio_enhancement:
            try:
                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": "Loading enhancement model...",
                        "progress": 16
                    })
                enhancer_model = await load_enhancement_model()
                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": "Applying audio enhancement...",
                        "progress": 18
                    })
                apply_enhancement(str(temp_wav), str(enhanced_wav), enhancer_model)
                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": "Enhancement complete",
                        "progress": 35
                    })
                temp_wav.unlink()
                temp_wav = enhanced_wav
            except Exception as exc:
                import logging
                logging.warning(f"Enhancement failed, using raw audio: {exc}")
                try:
                    enhanced_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                if progress_callback:
                    await progress_callback({
                        "stage": "enhancing",
                        "message": f"Enhancement skipped: {exc}",
                        "progress": 35
                    })
    # ---- End Stage 2 ----

    # ---- Stage 3: Load Model ----
    model = await load_model(model_key, progress_callback)

    # ---- Stage 3: Chunked Transcription ----
    if progress_callback:
        await progress_callback({
            "stage": "transcribing",
            "message": "Starting transcription...",
            "progress": 50
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

        if model_key == "qwen":
            chunk_text, chunk_lang, chunk_segments = await _transcribe_chunk_qwen(
                model, str(temp_wav), start_sec, end_sec
            )
        else:
            chunk_text, chunk_lang, chunk_segments = await _transcribe_chunk_whisper(
                model, str(temp_wav), start_sec, end_sec
            )

        if chunk_text.strip():
            all_texts.append(chunk_text)
        all_segments.extend(chunk_segments)
        if detected_language is None and chunk_lang:
            detected_language = chunk_lang

        chunk_progress = 50 + int((chunk_idx + 1) / total_chunks * 42)
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
            "progress": 92
        })

    timestamp = datetime.now().isoformat()
    safe_name = Path(original_filename).stem.replace(" ", "_")
    result_id = f"{safe_name}_{job_id[:8]}_{model_key}"

    srt_path = RESULTS_DIR / f"{result_id}.srt"
    srt_path.write_text(segments_to_srt(all_segments), encoding="utf-8")

    if progress_callback:
        await progress_callback({
            "stage": "saving",
            "message": "Results saved",
            "progress": 98
        })

    # ---- Cleanup ----
    try:
        temp_wav.unlink()
    except OSError:
        pass

    display_name = SUPPORTED_MODELS[model_key]["display_name"]

    if progress_callback:
        await progress_callback({
            "stage": "complete",
            "message": "Transcription complete",
            "progress": 100,
            "result_id": result_id,
            "filename": original_filename,
            "text": full_text,
            "language": detected_language,
            "model_key": model_key,
            "model_name": display_name,
        })

    return {
        "id": result_id,
        "filename": original_filename,
        "text": full_text,
        "language": detected_language,
        "timestamp": timestamp,
        "model_key": model_key,
        "model_name": display_name,
        "srt_path": str(srt_path),
    }


# === Qwen chunk transcription ===

async def _transcribe_chunk_qwen(
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


# === Whisper chunk transcription ===

async def _transcribe_chunk_whisper(
    model,
    wav_path: str,
    start_sec: float,
    end_sec: float,
) -> tuple[str, Optional[str], List[dict]]:
    def _run() -> tuple[str, Optional[str], List[dict]]:
        import whisper_timestamped as whisper

        chunk_id = uuid.uuid4().hex
        chunk_file = UPLOADS_DIR / f"chunk_{chunk_id}.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(start_sec), "-to", str(end_sec),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(chunk_file)
            ], capture_output=True, text=True)

            audio = whisper.load_audio(str(chunk_file))
            result = whisper.transcribe(model, audio, language=None)

            segments: List[dict] = []
            for seg in result.get("segments", []):
                seg_text = seg.get("text", "").strip()
                if seg_text:
                    segments.append({
                        "text": seg_text,
                        "start": start_sec + (seg.get("begin", 0) or 0) / 1000.0,
                        "end": start_sec + (seg.get("end", 0) or 0) / 1000.0,
                    })

            text = " ".join(s["text"] for s in segments)
            return text, result.get("language"), segments
        finally:
            try:
                chunk_file.unlink()
            except OSError:
                pass

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)
