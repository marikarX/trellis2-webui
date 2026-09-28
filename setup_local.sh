#!/usr/bin/env bash
# =============================================================================
# TRELLIS.2 WebUI — Local Development Setup (Full Build)
# =============================================================================
# Mirrors vastai_setup.sh for local workstations.
#
# Phase 1: Clone & build TRELLIS.2 (upstream) with all CUDA extensions
# Phase 2: Clone/link & install trellis2-webui on top
#
# Re-runnable: uses marker files to skip completed phases.
#
# Usage:
#   ./setup_local.sh                  # full build
#   ./setup_local.sh --skip-models    # skip HuggingFace model downloads
#   ./setup_local.sh --skip-frontend  # skip frontend npm build
#   ./setup_local.sh --skip-cuda-ext  # skip CUDA extension compilation
# =============================================================================
set -eo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via env vars
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRELLIS_REPO="${TRELLIS_REPO_URL:-https://github.com/microsoft/TRELLIS.2.git}"
TRELLIS_BRANCH="${TRELLIS_REPO_BRANCH:-main}"
TRELLIS_COMMIT="${TRELLIS_COMMIT:-5565d240c4a494caaf9ece7a554542b76ffa36d3}"  # pinned known-good

# Default: TRELLIS.2 as sibling directory
TRELLIS_DIR="${TRELLIS_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/TRELLIS.2}"
VENV_DIR="${TRELLIS_DIR}/.venv"

# Models
MODEL_BASE="${MODEL_BASE:-$HOME/AI/Models}"
MODEL_DIR="${MODEL_DIR:-${MODEL_BASE}/TRELLIS.2-4B}"
DINOV3_DIR="${DINOV3_DIR:-${MODEL_BASE}/dinov3-vitl16-pretrain-lvd1689m}"
HF_TOKEN="${HF_TOKEN:-}"

# Build config
EXTDIR="${TRELLIS_DIR}/.build_extensions"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PORT="${TRELLIS_PORT:-8000}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_MODELS=0
SKIP_FRONTEND=0
SKIP_CUDA_EXT=0

for arg in "$@"; do
    case "$arg" in
        --skip-models)    SKIP_MODELS=1 ;;
        --skip-frontend)  SKIP_FRONTEND=1 ;;
        --skip-cuda-ext)  SKIP_CUDA_EXT=1 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-models    Skip HuggingFace model downloads"
            echo "  --skip-frontend  Skip frontend npm build"
            echo "  --skip-cuda-ext  Skip CUDA extension compilation"
            echo ""
            echo "Environment overrides:"
            echo "  TRELLIS_DIR       Path to TRELLIS.2 clone (default: ../TRELLIS.2)"
            echo "  MODEL_BASE        Base models directory (default: ~/AI/Models)"
            echo "  MODEL_DIR         TRELLIS.2-4B model path"
            echo "  DINOV3_DIR        DINOv3 model path"
            echo "  MAX_JOBS          Parallel compile jobs (default: nproc)"
            echo "  PYTHON_VERSION    Python version for venv (default: 3.12)"
            echo "  HF_TOKEN          HuggingFace token for gated models"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE="${SCRIPT_DIR}/setup_local.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========================================"
echo "  TRELLIS.2 WebUI Local Setup — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================"
echo "  TRELLIS_DIR:    $TRELLIS_DIR"
echo "  MODEL_DIR:      $MODEL_DIR"
echo "  DINOV3_DIR:     $DINOV3_DIR"
echo "  PYTHON_VERSION: $PYTHON_VERSION"
echo "  MAX_JOBS:       $MAX_JOBS"
echo ""

# ---------------------------------------------------------------------------
# 1. System dependencies check
# ---------------------------------------------------------------------------
echo "[1/8] Checking system dependencies..."

MISSING=()
command -v git  &>/dev/null || MISSING+=("git")
command -v nvcc &>/dev/null || MISSING+=("cuda-toolkit (nvcc)")
dpkg -s libjpeg-dev &>/dev/null 2>&1 || MISSING+=("libjpeg-dev")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  Missing: ${MISSING[*]}"
    echo "  Install with: sudo apt install -y ${MISSING[*]}"
    echo ""
    echo "  Attempting to install..."
    sudo apt-get update -qq 2>&1 || true
    sudo apt-get install -y -qq libjpeg-dev libgl1 git 2>&1 || {
        echo "  ❌ Could not install system deps. Install manually and re-run."
        exit 1
    }
fi

if ! command -v nvcc &>/dev/null; then
    echo "  ❌ CUDA toolkit (nvcc) not found. Required for CUDA extension compilation."
    echo "  Install CUDA toolkit 13.x and ensure nvcc is on PATH."
    exit 1
fi

echo "  nvcc: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"

# ---------------------------------------------------------------------------
# 2. Install uv
# ---------------------------------------------------------------------------
echo "[2/8] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "  uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Clone TRELLIS.2 (upstream)
# ---------------------------------------------------------------------------
echo "[3/8] Setting up TRELLIS.2..."
if [ -d "${TRELLIS_DIR}/.git" ]; then
    echo "  Repo exists at ${TRELLIS_DIR}"
    pushd "$TRELLIS_DIR" > /dev/null
    git submodule update --init --recursive
    popd > /dev/null
else
    echo "  Cloning ${TRELLIS_REPO} (branch: ${TRELLIS_BRANCH})..."
    git clone -b "$TRELLIS_BRANCH" "$TRELLIS_REPO" "$TRELLIS_DIR" --recursive
    pushd "$TRELLIS_DIR" > /dev/null
    if [ -n "$TRELLIS_COMMIT" ]; then
        git checkout "$TRELLIS_COMMIT"
        echo "  Pinned to commit: ${TRELLIS_COMMIT:0:12}"
    fi
    popd > /dev/null
fi

# Patch rembg loading to be optional (BiRefNet has transformers version issues)
for pyfile in trellis2_image_to_3d.py trellis2_texturing.py; do
    python3 -c "
f = '$TRELLIS_DIR/trellis2/pipelines/$pyfile'
import os
if not os.path.exists(f):
    exit(0)
src = open(f).read()
old = \"pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\"
new = '''try:
            pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        except Exception as e:
            print(f'Warning: Background removal model failed to load: {e}')
            pipeline.rembg_model = None'''
if 'Warning: Background removal model failed to load' in src:
    print(f'  rembg already optional in $pyfile')
elif old in src:
    open(f, 'w').write(src.replace(old, new))
    print(f'  Patched rembg in $pyfile to be optional')
"
done

# ---------------------------------------------------------------------------
# 4. Phase 1: Build TRELLIS.2 venv + CUDA extensions
# ---------------------------------------------------------------------------
MARKER="${VENV_DIR}/.local_installed"
echo "[4/8] Phase 1 — Python environment + CUDA extensions..."

if [ -f "$MARKER" ]; then
    echo "  Already built (found marker). Skipping."
else
    pushd "$TRELLIS_DIR" > /dev/null

    # Create venv if missing
    if [ ! -d "$VENV_DIR" ]; then
        echo "  Creating venv (Python ${PYTHON_VERSION})..."
        uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    fi

    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="${VENV_DIR}/bin:$PATH"

    echo "  Installing PyTorch cu130..."
    uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

    echo "  Installing build tools..."
    uv pip install ninja packaging psutil setuptools wheel

    echo "  Installing basic dependencies..."
    uv pip install \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
        ninja trimesh transformers gradio==6.0.1 tensorboard pandas \
        lpips zstandard pillow-simd kornia timm huggingface-hub
    uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

    if [ "$SKIP_CUDA_EXT" = "1" ]; then
        echo "  ⏭️  --skip-cuda-ext: skipping CUDA extension compilation."
        echo "  You must build these manually: nvdiffrast, nvdiffrec, cumesh, flexgemm, o-voxel"
    else
        # -- CUDA extensions --
        echo "  Compiling CUDA extensions..."
        mkdir -p "$EXTDIR"

        # Auto-detect GPU compute capability
        GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
        if [ -n "$GPU_ARCH" ]; then
            export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
            echo "  Detected GPU arch: sm_${GPU_ARCH//./_} (compiling only for this)"
        fi
        export MAX_JOBS="$MAX_JOBS"
        echo "  Max parallel jobs: $MAX_JOBS"

        echo "    [1/5] nvdiffrast..."
        if [ ! -d "$EXTDIR/nvdiffrast" ]; then
            git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXTDIR/nvdiffrast" 2>/dev/null
        fi
        uv pip install "$EXTDIR/nvdiffrast" --no-build-isolation

        echo "    [2/5] nvdiffrec (renderutils)..."
        if [ ! -d "$EXTDIR/nvdiffrec" ]; then
            git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXTDIR/nvdiffrec" 2>/dev/null
        fi
        uv pip install "$EXTDIR/nvdiffrec" --no-build-isolation

        echo "    [3/5] CuMesh..."
        if [ ! -d "$EXTDIR/CuMesh" ]; then
            git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXTDIR/CuMesh" 2>/dev/null
            pushd "$EXTDIR/CuMesh" > /dev/null
            git checkout cf1a2f07304b5fe388ed86a16e4a0474599df914
            popd > /dev/null
        fi

        # Patch CuMesh for CUDA 13.2 CCCL/CUB compatibility
        # Old CUB: ExclusiveSum(temp, bytes, data, N) — 4-arg in-place
        # CUDA 13.2: ExclusiveSum(temp, bytes, d_in, d_out, N) — 5-arg separate in/out
        echo "      Patching CuMesh for CUDA 13.2 CUB compatibility..."
        CUMESH_PATCH="$EXTDIR/cumesh_cuda13_patch.py"
        cat > "$CUMESH_PATCH" << 'PYEOF'
"""Patch CuMesh for CUDA 13.2 CCCL/CUB compatibility.

Old CUB accepted 4-arg in-place ExclusiveSum/InclusiveSum:
    DeviceScan::ExclusiveSum(temp, bytes, data, N)
CUDA 13.2 requires 5-arg with separate in/out:
    DeviceScan::ExclusiveSum(temp, bytes, d_in, d_out, N)
"""
import os, sys, glob

def find_call_end(src, start):
    depth = 0
    i = start
    while i < len(src):
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1

def split_args(text):
    args = []
    depth = 0
    current = []
    for ch in text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args

def patch_file(fpath):
    with open(fpath) as f:
        src = f.read()
    orig = src
    fixes = 0

    for func_name in ["ExclusiveSum", "InclusiveSum"]:
        marker = f"cub::DeviceScan::{func_name}("
        offset = 0
        while True:
            idx = src.find(marker, offset)
            if idx == -1:
                break
            paren_start = idx + len(marker) - 1
            call_end = find_call_end(src, paren_start)
            if call_end == -1:
                offset = idx + 1
                continue
            inner = src[paren_start + 1 : call_end - 1]
            args = split_args(inner)
            if len(args) == 4:
                data_arg = args[2]
                count_arg = args[3]
                new_inner = f"{args[0]}, {args[1]},\n        {data_arg}, {data_arg},\n        static_cast<int>({count_arg})"
                src = src[:paren_start + 1] + new_inner + src[call_end - 1:]
                fixes += 1
                print(f"      Fixed {func_name} at offset {idx} ({len(args)} args -> 5 args)")
            offset = idx + 1

    if src != orig:
        with open(fpath, 'w') as f:
            f.write(src)
        return fixes
    return 0

src_dir = sys.argv[1]
total = 0
for fpath in glob.glob(os.path.join(src_dir, "src", "**", "*"), recursive=True):
    if fpath.endswith((".cu", ".h", ".cuh")):
        n = patch_file(fpath)
        if n:
            print(f"      Patched {os.path.basename(fpath)}: {n} fixes")
            total += n

for fpath in glob.glob(os.path.join(src_dir, "src", "*")):
    if fpath.endswith((".cu", ".h", ".cuh")):
        n = patch_file(fpath)
        if n:
            print(f"      Patched {os.path.basename(fpath)}: {n} fixes")
            total += n

print(f"      Total fixes applied: {total}")
PYEOF
        python3 "$CUMESH_PATCH" "$EXTDIR/CuMesh"

        # Remove .git so uv treats as plain local package (won't re-fetch ignoring patches)
        rm -rf "$EXTDIR/CuMesh/.git"
        uv pip install "$EXTDIR/CuMesh" --no-build-isolation

        echo "    [4/5] FlexGEMM..."
        if [ ! -d "$EXTDIR/FlexGEMM" ]; then
            git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXTDIR/FlexGEMM" 2>/dev/null
            pushd "$EXTDIR/FlexGEMM" > /dev/null
            git checkout 9f2f050396be3cc48894d15ce308e9672a07c677
            popd > /dev/null
        fi
        rm -rf "$EXTDIR/FlexGEMM/.git"
        uv pip install "$EXTDIR/FlexGEMM" --no-build-isolation

        echo "    [5/5] o-voxel..."
        if [ ! -d "$EXTDIR/o-voxel" ]; then
            cp -r "$TRELLIS_DIR/o-voxel" "$EXTDIR/o-voxel"
        fi
        # Remove git deps for cumesh/flex_gemm (already built from patched local clones)
        sed -i '/cumesh @/d; /flex_gemm @/d' "$EXTDIR/o-voxel/pyproject.toml"

        # Apply normal map bake patch to o-voxel source BEFORE pip install
        echo "      Applying normal map bake patch to o-voxel..."
        python3 "$SCRIPT_DIR/patches/normal_map_bake.py" "$EXTDIR/o-voxel"

        uv pip install "$EXTDIR/o-voxel" --no-build-isolation
    fi

    # -- NVIDIA runtime packages --
    echo "  Installing NVIDIA runtime packages..."
    uv pip install \
        nvidia-cublas-cu12 \
        nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 \
        nvidia-nvtx-cu12

    touch "$MARKER"
    echo "  ✅ Phase 1 complete."
    popd > /dev/null
fi

# Activate venv for remaining steps
export VIRTUAL_ENV="$VENV_DIR"
export PATH="${VENV_DIR}/bin:$PATH"

# ---------------------------------------------------------------------------
# 5. Install trellis2-webui into the shared venv
# ---------------------------------------------------------------------------
echo "[5/8] Installing trellis2-webui..."
pushd "$SCRIPT_DIR" > /dev/null

# Apply TRELLIS.2 patches from webui repo
echo "  Applying patches..."
python3 "$SCRIPT_DIR/patches/sparse_attn_sdpa.py" "$TRELLIS_DIR"
python3 "$SCRIPT_DIR/patches/dinov3_layer_compat.py" "$TRELLIS_DIR"
python3 "$SCRIPT_DIR/patches/decode_latent_oom_fix.py" "$TRELLIS_DIR"
python3 "$SCRIPT_DIR/patches/optional_rembg_preprocess_fallback.py" "$TRELLIS_DIR"

# Link example images from upstream
if [ ! -e "$SCRIPT_DIR/assets/example_image" ] && [ -d "$TRELLIS_DIR/assets/example_image" ]; then
    ln -s "$TRELLIS_DIR/assets/example_image" "$SCRIPT_DIR/assets/example_image"
    echo "  Linked example images from upstream"
fi

uv pip install -e . --no-build-isolation
echo "  ✅ WebUI installed."
popd > /dev/null

# ---------------------------------------------------------------------------
# 6. Download models
# ---------------------------------------------------------------------------
echo "[6/8] Checking models..."

if [ "$SKIP_MODELS" = "1" ]; then
    echo "  ⏭️  --skip-models: skipping HuggingFace downloads."
    echo "  Provide models manually:"
    echo "    TRELLIS.2-4B : ${MODEL_DIR}/"
    echo "    DINOv3       : ${DINOV3_DIR}/"
else
    # 6a. TRELLIS.2-4B
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/pipeline.json" ]; then
        echo "  TRELLIS.2-4B already downloaded."
    else
        echo "  Downloading TRELLIS.2-4B from HuggingFace (~16 GB)..."
        mkdir -p "$MODEL_DIR"
        python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B', local_dir='${MODEL_DIR}'${HF_TOKEN:+, token='${HF_TOKEN}'})"
        echo "  ✅ TRELLIS.2-4B downloaded."
    fi

    # 6b. DINOv3
    if [ -d "${DINOV3_DIR}" ] && [ -f "${DINOV3_DIR}/model.safetensors" ]; then
        echo "  DINOv3 already downloaded."
    else
        echo "  Downloading DINOv3 ViT-L/16 (~1.2 GB)..."
        mkdir -p "$DINOV3_DIR"
        python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', local_dir='${DINOV3_DIR}'${HF_TOKEN:+, token='${HF_TOKEN}'})"
        echo "  ✅ DINOv3 downloaded."
    fi
fi

# 6c. Patch pipeline configs for local DINOv3 path
if [ -f "${MODEL_DIR}/pipeline.json" ]; then
    echo "  Patching pipeline configs for DINOv3 path..."
    for cfg in "${MODEL_DIR}/pipeline.json" "${MODEL_DIR}/texturing_pipeline.json"; do
        if [ -f "$cfg" ]; then
            python3 -c "
import json
with open('$cfg') as f:
    data = json.load(f)
model_name = data.get('args', {}).get('image_cond_model', {}).get('args', {}).get('model_name', '')
if model_name != '${DINOV3_DIR}' and 'dinov3' in model_name.lower():
    data['args']['image_cond_model']['args']['model_name'] = '${DINOV3_DIR}'
    with open('$cfg', 'w') as f:
        json.dump(data, f, indent=4)
    print(f'  Patched {\"$cfg\".split(\"/\")[-1]}')
"
        fi
    done
else
    echo "  ⚠️  Models not found — config patching will happen on next run."
fi

# ---------------------------------------------------------------------------
# 7. Build frontend
# ---------------------------------------------------------------------------
echo "[7/8] Building frontend..."

if [ "$SKIP_FRONTEND" = "1" ]; then
    echo "  ⏭️  --skip-frontend: skipping npm build."
else
    # Check for Node.js
    if ! command -v node &>/dev/null; then
        echo "  ❌ Node.js not found. Install Node.js 18+ and re-run."
        echo "  Or use --skip-frontend if you don't need the UI."
        echo "  Install: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash - && sudo apt install -y nodejs"
    else
        echo "  Node.js: $(node --version)"
        pushd "${SCRIPT_DIR}/webui" > /dev/null
        if [ -d "dist" ] && [ "dist/index.html" -nt "index.html" ]; then
            echo "  Frontend already built."
        else
            npm ci --silent 2>/dev/null || npm install --silent
            npm run build
            echo "  ✅ Frontend built."
        fi
        popd > /dev/null
    fi
fi

# ---------------------------------------------------------------------------
# 8. Verification
# ---------------------------------------------------------------------------
echo "[8/8] Verifying installation..."

PYTHONPATH="${TRELLIS_DIR}:${PYTHONPATH:-}" python -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')
try:
    import o_voxel, cumesh, flex_gemm, nvdiffrast, nvdiffrec_render, utils3d
    print('  All CUDA extensions OK')
except ImportError as e:
    print(f'  ⚠️  Missing extension: {e}')
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print('  TRELLIS.2 pipeline import OK')
except ImportError as e:
    print(f'  ⚠️  Pipeline import failed: {e}')
"

echo ""
echo "========================================"
echo "  ✅ Local setup complete!"
echo "========================================"
echo ""
echo "  To run the WebUI:"
echo ""
echo "    source ${VENV_DIR}/bin/activate"
echo "    export TRELLIS_MODEL_PATH=${MODEL_DIR}"
echo "    cd ${SCRIPT_DIR}"
echo "    python app.py"
echo ""
echo "  Or use the convenience script:"
echo "    ./run_local.sh"
echo ""
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================"
