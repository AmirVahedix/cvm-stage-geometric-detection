#!/usr/bin/env bash
# ==============================================================================
# Cloud GPU Runner for CVM Landmark Inference & 4-Run Stage Classification
# Compatible with: Vast.ai, RunPod, Lambda Labs, AWS EC2, GCP, Ubuntu instances
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "🚀 Initializing CVM Multi-Run Evaluation on Cloud GPU"
echo "=================================================================="

# 1. Verify NVIDIA GPU & Driver
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠️ WARNING: nvidia-smi not found. Ensure an NVIDIA GPU is attached."
fi

# 2. System dependencies (for OpenCV / PIL / Headless environments)
if command -v apt-get &> /dev/null; then
    echo "📦 Checking system libraries..."
    apt-get update -qq && apt-get install -y -qq \
        libgl1 \
        libglib2.0-0 \
        python3-pip \
        python3-venv \
        unzip \
        rsync \
        curl \
        git > /dev/null 2>&1 || true
fi

# 3. Setup Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# 4. Install / Verify PyTorch with CUDA support
echo "⚡ Checking PyTorch CUDA acceleration..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in current torch'" 2>/dev/null || {
    echo "📥 Installing PyTorch with CUDA 12.1..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

# 5. Install project dependencies
echo "📦 Installing project dependencies..."
pip install -q \
    timm \
    albumentations \
    pillow \
    tqdm \
    python-dotenv \
    requests \
    numpy \
    matplotlib \
    scipy

# 6. Verify Model Weights and Dataset
if [ ! -f "model/weights.pth" ]; then
    echo "❌ Error: Model weights not found at model/weights.pth"
    echo "   Please upload 'model/weights.pth' from your local machine."
    exit 1
fi

if [ ! -d "data/full/images" ]; then
    if [ -f "data/full.zip" ]; then
        echo "📦 Extracting dataset from data/full.zip into data/..."
        unzip -q data/full.zip -d data/
    else
        echo "❌ Error: Images directory not found at data/full/images"
        echo "   Please upload 'data/full.zip' or the 'data/full' folder."
        exit 1
    fi
fi

if [ ! -f "data/export_cache.json" ]; then
    echo "❌ Error: Label Studio export not found at data/export_cache.json"
    echo "   Please upload 'data/export_cache.json' from your local machine."
    exit 1
fi

# 7. Print GPU info from PyTorch
python3 -c "
import torch
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA Available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ Active GPU     :', torch.cuda.get_device_name(0))
    print('✅ Device Count   :', torch.cuda.device_count())
"

# 8. Run full inference & 4 CVM calculator runs on CUDA
echo "=================================================================="
echo "🚀 Starting Full 4-Run Evaluation on CUDA GPU..."
echo "=================================================================="

python3 inference_on_label_studio_project_1.py \
    --device cuda \
    --sort-order ordinal \
    --stage-format int \
    --output-dir . \
    --run1-output run1_gt_standard.txt \
    --run2-output run2_pred_standard.txt \
    --run3-output run3_gt_fuzzy.txt \
    --run4-output run4_pred_fuzzy.txt \
    --summary-json runs_distribution.json \
    --save-mapping runs_comparison.csv \
    "$@"

echo "=================================================================="
echo "🎉 Completed successfully! Outputs saved:"
echo "   - run1_gt_standard.txt   (Run 1: Standard on Ground-Truth)"
echo "   - run2_pred_standard.txt (Run 2: Standard on Predicted)"
echo "   - run3_gt_fuzzy.txt      (Run 3: Fuzzy on Ground-Truth)"
echo "   - run4_pred_fuzzy.txt    (Run 4: Fuzzy on Predicted)"
echo "   - runs_distribution.json (6-class distributions & accuracy)"
echo "   - runs_comparison.csv    (Ordinal per-image comparison table)"
echo "=================================================================="
