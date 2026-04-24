import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "uploads"

HOST = os.getenv("STT_HOST", "0.0.0.0")
PORT = int(os.getenv("STT_PORT", "8000"))

MODEL_NAME = os.getenv("STT_MODEL", "Qwen/Qwen3-ASR-1.7B")
FORCE_ALIGNER_NAME = os.getenv("STT_FORCE_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")
DTYPE = os.getenv("STT_DTYPE", "bfloat16")
CHUNK_DURATION_SEC = int(os.getenv("STT_CHUNK_DURATION", "30"))
MAX_NEW_TOKENS = int(os.getenv("STT_MAX_NEW_TOKENS", "256"))

MAX_FILE_SIZE_MB = int(os.getenv("STT_MAX_FILE_MB", "10000"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ENABLE_NOISE_REDUCTION = os.getenv("STT_ENABLE_NOISE_REDUCTION", "true").lower() in ("true", "1", "yes")
ENABLE_AUDIO_ENHANCEMENT = os.getenv("STT_ENABLE_AUDIO_ENHANCEMENT", "true").lower() in ("true", "1", "yes")
NOISE_REDUCTION_MODEL = os.getenv("STT_NOISE_REDUCTION_MODEL", "JacobLinCool/MP-SENet-DNS")
ENHANCEMENT_MODEL = os.getenv("STT_ENHANCEMENT_MODEL", "speechbrain/metricgan-plus-voicebank")

for d in [RESULTS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
