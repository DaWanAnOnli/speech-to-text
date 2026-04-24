#!/bin/bash
set -e

echo "=== Speech-to-Text Setup ==="

# 1. System dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg python3.12 python3.12-venv python3-pip git curl

# 2. Python virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. PyTorch with CUDA 12.8 (supports RTX 5090 / Blackwell / sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5. Install qwen-asr (pulls transformers, accelerate, etc. as transitive deps)
pip install qwen-asr

# 6. Install whisper-timestamped (for Whisper-large-v3 transcription with timestamps)
pip install whisper-timestamped

# 7. Install FastAPI and remaining Python deps
pip install -r requirements.txt

# 8. Create working directories
mkdir -p results uploads

# 9. Pre-download the models to cache (~10-12GB total from HuggingFace)
# Qwen3-ASR-1.7B + ForcedAligner (~4.6GB)
# Loading on CPU just to cache weights; GPU loading happens at runtime
# Qwen3-ASR-1.7B + ForcedAligner (~4.6GB)
python -c "from qwen_asr import Qwen3ASRModel; import torch; Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B', dtype=torch.float32, device_map='cpu', forced_aligner='Qwen/Qwen3-ForcedAligner-0.6B', forced_aligner_kwargs=dict(dtype=torch.float32, device_map='cpu'))"
# Whisper-large-v3 (~3GB)
python -c "import whisper_timestamped; whisper_timestamped.load_model('large-v3', device='cpu')"

# 10. Pre-download enhancement models to cache (~3-5GB total)
# Sepformer noise reduction via SpeechBrain
python -c "
from speechbrain.pretrained import SepformerSeparation
model = SepformerSeparation.from_hparams(
    source='speechbrain/sepformer-wham16k-enhancement',
    run_opts={'device': 'cpu'}
)
print('Sepformer noise reduction cached')
"

# MetricGAN+ via SpeechBrain
python -c "
from speechbrain.inference.enhancement import SpectralMaskEnhancement
model = SpectralMaskEnhancement.from_hparams(
    source='speechbrain/metricgan-plus-voicebank',
    run_opts={'device': 'cpu'}
)
print('MetricGAN+ cached')
"

echo "Setup complete."
echo "Activate venv: source venv/bin/activate"
echo "Start server: python main.py"
echo ""
echo "Optional environment variables:"
echo "  STT_MAX_CONCURRENT_JOBS  Max simultaneous GPU jobs (default: 2, set to 1 for max safety)"
