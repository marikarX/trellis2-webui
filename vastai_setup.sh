#!/usr/bin/env bash
# =============================================================================
# TRELLIS.2 WebUI — Vast.ai Provisioning Script (Two-Phase Install)
# =============================================================================
# Usage: Set this script's raw GitHub URL as the PROVISIONING_SCRIPT env var
#        in your vast.ai template.  Image: vastai/base-image:cuda-13.2.0-auto
#
# Phase 1: Clone & build TRELLIS.2 (upstream) with all CUDA extensions
# Phase 2: Clone & install trellis2-webui (this repo) on top
#
# The WebUI is registered as a Supervisor service so it auto-restarts,
# logs are accessible, and it coexists with Jupyter/SSH/Syncthing.
#
# All state lives under /workspace/ (persists across stop/start).
# First boot: ~15-25 min. Subsequent boots: ~30 s.
# =============================================================================
set -eo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via vast.ai env vars
# ---------------------------------------------------------------------------
TRELLIS_REPO="${TRELLIS_REPO_URL:-https://github.com/microsoft/TRELLIS.2.git}"
TRELLIS_BRANCH="${TRELLIS_REPO_BRANCH:-main}"
TRELLIS_COMMIT="${TRELLIS_COMMIT:-5565d240c4a494caaf9ece7a554542b76ffa36d3}"  # pinned known-good
WEBUI_REPO="${WEBUI_REPO_URL:-https://github.com/matatratata/trellis2-webui.git}"
WEBUI_BRANCH="${WEBUI_REPO_BRANCH:-main}"

TRELLIS_DIR="/workspace/TRELLIS.2"
WEBUI_DIR="/workspace/trellis2-webui"
MODEL_DIR="/workspace/models/TRELLIS.2-4B"
VENV_DIR="${TRELLIS_DIR}/.venv"
LOG_FILE="/workspace/setup.log"
PORT="${TRELLIS_PORT:-8000}"

HF_TOKEN="${HF_TOKEN:-}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"  # Set to 1 to skip HF model downloads

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_apt() {
    local tries=0
    while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
          fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
        if [ $tries -eq 0 ]; then
            echo "  Waiting for apt lock (base image still running)..."
        fi
        tries=$((tries + 1))
        sleep 2
        if [ $tries -ge 30 ]; then
            echo "  WARNING: apt lock held for 60 s, proceeding anyway"
            break
        fi
    done
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========================================"
echo "  TRELLIS.2 WebUI Setup — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/9] Installing system packages..."
wait_for_apt
apt-get update -qq 2>&1 || echo "  WARNING: apt-get update had issues, continuing..."
apt-get install -y -qq libjpeg-dev libgl1 git curl 2>&1 || echo "  WARNING: some apt packages may have failed"

if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null 2>&1
fi
echo "  Node.js: $(node --version)"

# ---------------------------------------------------------------------------
# 2. Install uv
# ---------------------------------------------------------------------------
echo "[2/9] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "  uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Clone TRELLIS.2 (upstream)
# ---------------------------------------------------------------------------
echo "[3/9] Setting up TRELLIS.2..."
if [ -d "${TRELLIS_DIR}/.git" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$TRELLIS_DIR"
    git pull --ff-only || true
    git submodule update --init --recursive
else
    echo "  Cloning ${TRELLIS_REPO} (branch: ${TRELLIS_BRANCH})..."
    git clone -b "$TRELLIS_BRANCH" "$TRELLIS_REPO" "$TRELLIS_DIR" --recursive
    cd "$TRELLIS_DIR"
    git checkout "$TRELLIS_COMMIT"  # pin to known-good commit
fi
# Patch rembg loading to be optional (BiRefNet has transformers version issues)
# Applies to BOTH the main pipeline and the texturing pipeline
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
if old in src:
    open(f, 'w').write(src.replace(old, new))
    print(f'  Patched rembg in {\"$pyfile\"} to be optional')
"
done

# ---------------------------------------------------------------------------
# 4. Phase 1: Build TRELLIS.2 venv + CUDA extensions
# ---------------------------------------------------------------------------
MARKER="${VENV_DIR}/.vastai_installed"
echo "[4/9] Phase 1 — Python environment + CUDA extensions..."

if [ -f "$MARKER" ]; then
    echo "  Already built (found marker). Skipping."
else
    cd "$TRELLIS_DIR"
    uv venv --python 3.10 --clear "$VENV_DIR"

    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="${VENV_DIR}/bin:$PATH"

    echo "  Installing PyTorch cu130..."
    uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

    echo "  Installing build tools..."
    uv pip install ninja packaging psutil setuptools wheel

    # -- Basic dependencies (mirrors setup.sh --basic) --
    echo "  Installing basic dependencies..."
    uv pip install \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
        ninja trimesh transformers gradio==6.0.1 tensorboard pandas \
        lpips zstandard pillow-simd kornia timm huggingface-hub
    uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

    # -- CUDA extensions (mirrors setup.sh flags) --
    echo "  Compiling CUDA extensions (~10-15 min)..."
    EXTDIR="/tmp/extensions"
    mkdir -p "$EXTDIR"

    # Auto-detect GPU compute capability and limit parallel jobs
    GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
    if [ -n "$GPU_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
        echo "  Detected GPU arch: sm_${GPU_ARCH//./_} (compiling only for this)"
    fi
    export MAX_JOBS="${MAX_JOBS:-2}"
    echo "  Max parallel jobs: $MAX_JOBS"

    echo "    [1/5] nvdiffrast..."
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXTDIR/nvdiffrast" 2>/dev/null || true
    uv pip install "$EXTDIR/nvdiffrast" --no-build-isolation

    echo "    [2/5] nvdiffrec (renderutils)..."
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXTDIR/nvdiffrec" 2>/dev/null || true
    uv pip install "$EXTDIR/nvdiffrec" --no-build-isolation

    echo "    [3/5] CuMesh..."
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXTDIR/CuMesh" 2>/dev/null || true
    cd "$EXTDIR/CuMesh" && git checkout cf1a2f07304b5fe388ed86a16e4a0474599df914 && cd -

    # Patch ALL CuMesh sources for CUDA 13.2 CCCL/CUB compatibility.
    # Old CUB accepted in-place 4-arg ExclusiveSum/InclusiveSum:
    #   ExclusiveSum(temp, bytes, data, N)
    # CUDA 13.2 requires 5-arg with separate in/out:
    #   ExclusiveSum(temp, bytes, d_in, d_out, N)
    # This affects shared.h (compress_ids) and clean_up.cu (remove_unreferenced_vertices, fill_holes)
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
    """Find matching closing paren, returning index after ')'."""
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
    """Split text by top-level commas (respecting nested parens)."""
    args = []
    depth = 0
    current = []
    for ch in text:
        if ch == '(' :
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
            # Find the opening paren
            paren_start = idx + len(marker) - 1  # index of '('
            call_end = find_call_end(src, paren_start)
            if call_end == -1:
                offset = idx + 1
                continue

            # Extract args between parens
            inner = src[paren_start + 1 : call_end - 1]
            args = split_args(inner)

            if len(args) == 4:
                # 4-arg in-place call -> 5-arg with separate in/out
                # args: [temp_storage, temp_bytes, data, count]
                # becomes: [temp_storage, temp_bytes, data, data, (int)count]
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

# Also check files directly in src/ (not just subdirectories)
for fpath in glob.glob(os.path.join(src_dir, "src", "*")):
    if fpath.endswith((".cu", ".h", ".cuh")):
        n = patch_file(fpath)
        if n:
            print(f"      Patched {os.path.basename(fpath)}: {n} fixes")
            total += n

print(f"      Total fixes applied: {total}")
PYEOF
    python3 "$CUMESH_PATCH" "$EXTDIR/CuMesh"

    # Remove .git so uv treats this as a plain local package, not a VCS source.
    # Without this, uv re-fetches a clean checkout into its cache and ignores our patches!
    rm -rf "$EXTDIR/CuMesh/.git"

    uv pip install "$EXTDIR/CuMesh" --no-build-isolation

    echo "    [4/5] FlexGEMM..."
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXTDIR/FlexGEMM" 2>/dev/null || true
    cd "$EXTDIR/FlexGEMM" && git checkout 9f2f050396be3cc48894d15ce308e9672a07c677 && cd -
    rm -rf "$EXTDIR/FlexGEMM/.git"
    uv pip install "$EXTDIR/FlexGEMM" --no-build-isolation

    echo "    [5/5] o-voxel..."
    cp -r "$TRELLIS_DIR/o-voxel" "$EXTDIR/o-voxel" 2>/dev/null || true
    # Remove git deps for cumesh/flex_gemm from o-voxel's pyproject.toml
    # (we already built and installed them from patched local clones)
    sed -i '/cumesh @/d; /flex_gemm @/d' "$EXTDIR/o-voxel/pyproject.toml"

    # Patch o-voxel postprocess.py to bake tangent-space normal maps.
    # Must be inlined here because $WEBUI_DIR hasn't been cloned yet (Phase 1).
    echo "      Patching postprocess.py for tangent-space normal map baking..."
    NMAP_PATCH="$EXTDIR/normal_map_bake_patch.py"
    cat > "$NMAP_PATCH" << 'PYEOF'
"""
Patch: Bake TANGENT-SPACE normal map in o_voxel.postprocess.to_glb()

Replaces analytically flat derivatives with smooth vertex tangents (MikkTSpace)
and fixes random dual-contouring winding noise using hemisphere alignment.

Improvements over v1:
  - Angle-weighted tangent accumulation (matches MikkTSpace weighting)
  - Iterative dilation + inpaint for robust seam padding
"""
import sys
import os
import re

def patch(ovoxel_dir):
    target = os.path.join(ovoxel_dir, 'o_voxel', 'postprocess.py')
    if not os.path.exists(target):
        print(f"  [SKIP] {target} not found")
        return
    src = open(target).read()
    changed = False

    # 1. Unify High-Res Mesh Winding (Fixes normal cancellation / rainbow noise)
    fill_holes_pattern = r"(mesh\.fill_holes\(max_hole_perimeter=3e-2\)\n)(.*?vertices, faces = mesh\.read\(\))"
    new_fill_holes = r"\1    mesh.remove_duplicate_faces()\n    mesh.repair_non_manifold_edges()\n    mesh.unify_face_orientations()\n\2"
    if "mesh.unify_face_orientations()" not in src[:src.find("vertices, faces = mesh.read()")]:
        src = re.sub(fill_holes_pattern, new_fill_holes, src, flags=re.DOTALL)
        changed = True

    # 2. Ensure simplified mesh is unified before unwrapping
    unwrap_pattern = r"(\s+)(out_vertices, out_faces, out_uvs, out_vmaps = mesh\.uv_unwrap\()"
    new_unwrap = r"\1mesh.unify_face_orientations()\n\1\2"
    if "mesh.unify_face_orientations()\n    out_vertices, out_faces" not in src:
        src = re.sub(unwrap_pattern, new_unwrap, src)
        changed = True

    # 3. Clean up the old rast_db tracking that ate VRAM and caused flat polygons
    if "rast_deriv" in src:
        src = re.sub(r"    rast_deriv = torch\.zeros.*?\n", "", src)
        src = src.replace("rast_chunk, rast_db_chunk = dr.rasterize(", "rast_chunk, _ = dr.rasterize(")
        src = re.sub(r"        rast_deriv = torch\.where\(mask_chunk, rast_db_chunk, rast_deriv\)\n", "", src)
        changed = True

    # 4. Smooth Tangent-Space Baking Block (angle-weighted MikkTSpace tangents)
    tangent_space_bake = r"""\1
    # --- Bake TANGENT-SPACE normal map (Smooth MikkTSpace-style, angle-weighted) ---
    _v0, _v1, _v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    _fn = torch.cross(_v1 - _v0, _v2 - _v0, dim=-1)
    orig_vnormals = torch.zeros_like(vertices)
    orig_vnormals.scatter_add_(0, faces[:, 0].unsqueeze(1).expand(-1, 3), _fn)
    orig_vnormals.scatter_add_(0, faces[:, 1].unsqueeze(1).expand(-1, 3), _fn)
    orig_vnormals.scatter_add_(0, faces[:, 2].unsqueeze(1).expand(-1, 3), _fn)
    orig_vnormals = orig_vnormals / orig_vnormals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    del _v0, _v1, _v2, _fn
    
    _tri_nrm = orig_vnormals[faces[face_id.long()]]
    hires_valid = (_tri_nrm * uvw.unsqueeze(-1)).sum(dim=1)
    hires_valid = hires_valid / hires_valid.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    del _tri_nrm, orig_vnormals
    
    hires_nrm = torch.zeros(texture_size, texture_size, 3, device='cuda', dtype=torch.float32)
    hires_nrm[mask] = hires_valid
    del hires_valid
    
    # Calculate angle-weighted smooth tangents at vertices (MikkTSpace-style)
    _v0, _v1, _v2 = out_vertices[out_faces[:, 0]], out_vertices[out_faces[:, 1]], out_vertices[out_faces[:, 2]]
    _uv0, _uv1, _uv2 = out_uvs[out_faces[:, 0]], out_uvs[out_faces[:, 1]], out_uvs[out_faces[:, 2]]
    dp1, dp2 = _v1 - _v0, _v2 - _v0
    duv1, duv2 = _uv1 - _uv0, _uv2 - _uv0
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    r = torch.where(det.abs() > 1e-8, 1.0 / det, torch.zeros_like(det))
    face_T = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * r.unsqueeze(1)
    face_B = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * r.unsqueeze(1)
    # Angle weights per corner vertex (MikkTSpace uses angle, not area)
    _e01, _e02 = _v1 - _v0, _v2 - _v0
    _e12, _e10 = _v2 - _v1, _v0 - _v1
    _e20, _e21 = _v0 - _v2, _v1 - _v2
    _cos0 = (_e01 * _e02).sum(-1) / (_e01.norm(dim=-1) * _e02.norm(dim=-1)).clamp(min=1e-8)
    _cos1 = (_e12 * _e10).sum(-1) / (_e12.norm(dim=-1) * _e10.norm(dim=-1)).clamp(min=1e-8)
    _cos2 = (_e20 * _e21).sum(-1) / (_e20.norm(dim=-1) * _e21.norm(dim=-1)).clamp(min=1e-8)
    _ang0 = torch.acos(_cos0.clamp(-1.0, 1.0)).unsqueeze(1)
    _ang1 = torch.acos(_cos1.clamp(-1.0, 1.0)).unsqueeze(1)
    _ang2 = torch.acos(_cos2.clamp(-1.0, 1.0)).unsqueeze(1)
    del _e01, _e02, _e12, _e10, _e20, _e21, _cos0, _cos1, _cos2
    del _v0, _v1, _v2, _uv0, _uv1, _uv2, dp1, dp2, duv1, duv2, det, r
    
    out_T, out_B = torch.zeros_like(out_vertices), torch.zeros_like(out_vertices)
    out_T.scatter_add_(0, out_faces[:, 0].unsqueeze(1).expand(-1, 3), face_T * _ang0)
    out_T.scatter_add_(0, out_faces[:, 1].unsqueeze(1).expand(-1, 3), face_T * _ang1)
    out_T.scatter_add_(0, out_faces[:, 2].unsqueeze(1).expand(-1, 3), face_T * _ang2)
    out_B.scatter_add_(0, out_faces[:, 0].unsqueeze(1).expand(-1, 3), face_B * _ang0)
    out_B.scatter_add_(0, out_faces[:, 1].unsqueeze(1).expand(-1, 3), face_B * _ang1)
    out_B.scatter_add_(0, out_faces[:, 2].unsqueeze(1).expand(-1, 3), face_B * _ang2)
    del face_T, face_B, _ang0, _ang1, _ang2
    out_T = out_T / out_T.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    out_B = out_B / out_B.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Interpolate smooth frame at each texel
    simp_N = dr.interpolate(out_normals.unsqueeze(0), rast, out_faces)[0][0]
    simp_T = dr.interpolate(out_T.unsqueeze(0), rast, out_faces)[0][0]
    simp_B = dr.interpolate(out_B.unsqueeze(0), rast, out_faces)[0][0]
    del out_T, out_B
    simp_N = simp_N / simp_N.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Safe Gram-Schmidt Orthogonalization
    T = simp_T - (simp_T * simp_N).sum(-1, keepdim=True) * simp_N
    T_norm = T.norm(dim=-1, keepdim=True)
    T = torch.where(T_norm > 1e-8, T / T_norm, torch.tensor([1.0, 0.0, 0.0], device='cuda', dtype=torch.float32))
    B = simp_B - (simp_B * simp_N).sum(-1, keepdim=True) * simp_N - (simp_B * T).sum(-1, keepdim=True) * T
    B_norm = B.norm(dim=-1, keepdim=True)
    B = torch.where(B_norm > 1e-8, B / B_norm, torch.cross(simp_N, T, dim=-1))
    
    # Hemispherical alignment: forces high-res normal to point outwards
    flip = (hires_nrm * simp_N).sum(dim=-1, keepdim=True).sign()
    flip = torch.where(flip == 0, torch.ones_like(flip), flip)
    hires_nrm = hires_nrm * flip
    
    # Map to tangent space (negating B compensates for the UV V-flip during export)
    ts = torch.stack([(hires_nrm * T).sum(-1), -(hires_nrm * B).sum(-1), (hires_nrm * simp_N).sum(-1)], dim=-1)
    ts = ts / ts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ts[~mask] = torch.tensor([0.0, 0.0, 1.0], device='cuda', dtype=torch.float32)
    normal_map_np = np.clip((ts.cpu().numpy() * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    \2"""

    if "# --- Bake TANGENT-SPACE normal map (Smooth MikkTSpace-style" not in src:
        # Replaces everything from the trilinear grid_sample output to the PBR construction
        bake_pattern = r"(mode='trilinear',\n\s+\)\n).*?(    # --- Texture Post-Processing & Material Construction ---)"
        new_src = re.sub(bake_pattern, tangent_space_bake, src, flags=re.DOTALL)
        if new_src != src:
            src = new_src
            changed = True
            print("  Patched to_glb: upgraded to robust smooth tangent-space baking")

    # 5. Iterative dilation + Inpaint Normal Map + Add to Material
    inpaint_pattern = r"(alpha = cv2\.inpaint.*?)\n(\s+# Create PBR material)"
    new_inpaint = (
        r"\1\n"
        r"    normal_map_np[~mask] = [128, 128, 255]\n"
        r"    _nmap_orig = normal_map_np.copy()\n"
        r"    _dilate_kern = np.ones((3, 3), dtype=np.uint8)\n"
        r"    for _ in range(4):\n"
        r"        normal_map_np = cv2.dilate(normal_map_np, _dilate_kern, iterations=1)\n"
        r"        normal_map_np[mask] = _nmap_orig[mask]\n"
        r"    normal_map_np = cv2.inpaint(normal_map_np, (~mask & (normal_map_np[..., 2] == 255)).astype(np.uint8), 3, cv2.INPAINT_TELEA)\n"
        r"    del _nmap_orig, _dilate_kern\n"
        r"\2"
    )
    if "normal_map_np[~mask] = [128, 128, 255]" not in src:
        src = re.sub(inpaint_pattern, new_inpaint, src)
        changed = True

    material_pattern = r"(roughnessFactor=1\.0,\n)(\s+)(alphaMode=alpha_mode,)"
    new_material = r"\1\2normalTexture=Image.fromarray(normal_map_np),\n\2\3"
    if "normalTexture=Image.fromarray(normal_map_np)" not in src:
        src = re.sub(material_pattern, new_material, src)
        changed = True

    if changed:
        open(target, 'w').write(src)
        print("    Tangent-space normal map patches successfully updated in postprocess.py")
    else:
        print("  No changes made (already up to date).")

if __name__ == '__main__':
    ovoxel_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/extensions/o-voxel'
    patch(ovoxel_dir)
PYEOF
    python3 "$NMAP_PATCH" "$EXTDIR/o-voxel"

    uv pip install "$EXTDIR/o-voxel" --no-build-isolation

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
fi

# Activate venv for remaining steps
export VIRTUAL_ENV="$VENV_DIR"
export PATH="${VENV_DIR}/bin:$PATH"

# ---------------------------------------------------------------------------
# 5. Clone trellis2-webui (this repo)
# ---------------------------------------------------------------------------
echo "[5/9] Setting up trellis2-webui..."
if [ -d "${WEBUI_DIR}/.git" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$WEBUI_DIR"
    git pull --ff-only || true
else
    echo "  Cloning ${WEBUI_REPO} (branch: ${WEBUI_BRANCH})..."
    git clone -b "$WEBUI_BRANCH" "$WEBUI_REPO" "$WEBUI_DIR"
    cd "$WEBUI_DIR"
fi

# Apply TRELLIS.2 patches from webui repo
echo "  Applying patches..."
python3 "$WEBUI_DIR/patches/sparse_attn_sdpa.py" "$TRELLIS_DIR"
python3 "$WEBUI_DIR/patches/dinov3_layer_compat.py" "$TRELLIS_DIR"
python3 "$WEBUI_DIR/patches/decode_latent_oom_fix.py" "$TRELLIS_DIR"
python3 "$WEBUI_DIR/patches/optional_rembg_preprocess_fallback.py" "$TRELLIS_DIR"

# Link example images from upstream TRELLIS.2
if [ ! -e "$WEBUI_DIR/assets/example_image" ] && [ -d "$TRELLIS_DIR/assets/example_image" ]; then
    ln -s "$TRELLIS_DIR/assets/example_image" "$WEBUI_DIR/assets/example_image"
    echo "  Linked example images from upstream"
fi

# ---------------------------------------------------------------------------
# 6. Phase 2: Install webui dependencies into the shared venv
# ---------------------------------------------------------------------------
echo "[6/9] Phase 2 — Installing WebUI dependencies..."
cd "$WEBUI_DIR"
uv pip install -e .

# ---------------------------------------------------------------------------
# 7. Download models
# ---------------------------------------------------------------------------
echo "[7/9] Checking models..."

DINOV3_DIR="/workspace/models/dinov3-vitl16-pretrain-lvd1689m"

if [ "$SKIP_MODEL_DOWNLOAD" = "1" ]; then
    echo "  ⏭️  SKIP_MODEL_DOWNLOAD=1 — skipping HuggingFace downloads."
    echo "  Provide models manually to these paths:"
    echo "    TRELLIS.2-4B : ${MODEL_DIR}/"
    echo "    DINOv3       : ${DINOV3_DIR}/"
    echo ""
    echo "  Example (from another machine):"
    echo "    rsync -avP TRELLIS.2-4B/ root@<host>:${MODEL_DIR}/"
    echo "    rsync -avP dinov3-vitl16-pretrain-lvd1689m/ root@<host>:${DINOV3_DIR}/"
else
    # 7a. TRELLIS.2-4B main model
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
        echo "  TRELLIS.2-4B already downloaded."
    else
        echo "  Downloading TRELLIS.2-4B from HuggingFace (~16 GB)..."
        python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B', local_dir='${MODEL_DIR}'${HF_TOKEN:+, token='${HF_TOKEN}'})"
        echo "  ✅ TRELLIS.2-4B downloaded."
    fi

    # 7b. DINOv3 vision encoder (needed by both main + texturing pipelines)
    if [ -d "${DINOV3_DIR}" ] && [ -f "${DINOV3_DIR}/model.safetensors" ]; then
        echo "  DINOv3 already downloaded."
    else
        echo "  Downloading DINOv3 ViT-L/16 (~1.2 GB)..."
        python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', local_dir='${DINOV3_DIR}'${HF_TOKEN:+, token='${HF_TOKEN}'})"
        echo "  ✅ DINOv3 downloaded."
    fi
fi

# 7c. Patch pipeline configs to use the Vast.ai model path for DINOv3
# (runs regardless — patches configs if models were provided manually)
if [ -f "${MODEL_DIR}/config.json" ]; then
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
    echo "  ⚠️  Models not found yet — config patching will happen on next run."
fi

# ---------------------------------------------------------------------------
# 8. Build frontend
# ---------------------------------------------------------------------------
echo "[8/9] Building frontend..."
cd "${WEBUI_DIR}/webui"
if [ -d "dist" ] && [ "dist/index.html" -nt "index.html" ]; then
    echo "  Frontend already built."
else
    npm ci --silent 2>/dev/null || npm install --silent
    npm run build
    echo "  ✅ Frontend built."
fi

# ---------------------------------------------------------------------------
# 9. Register WebUI as Supervisor service
# ---------------------------------------------------------------------------
echo "[9/9] Registering WebUI as Supervisor service..."

SUPERVISOR_CONF="/etc/supervisor/conf.d/trellis-webui.conf"
WEBUI_LAUNCHER="/opt/supervisor-scripts/trellis-webui.sh"

# Create launcher script
mkdir -p /opt/supervisor-scripts
cat > "$WEBUI_LAUNCHER" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
set -euo pipefail

# Activate the TRELLIS venv
export VIRTUAL_ENV="/workspace/TRELLIS.2/.venv"
export PATH="${VIRTUAL_ENV}/bin:$PATH"

# Environment
export TRELLIS_MODEL_PATH="/workspace/models/TRELLIS.2-4B"
export PYTHONPATH="/workspace/TRELLIS.2:${PYTHONPATH:-}"
export ATTN_BACKEND="sdpa"
export SPARSE_ATTN_BACKEND="sdpa"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OPENCV_IO_ENABLE_OPENEXR="1"

PORT="${TRELLIS_PORT:-8000}"

# Patch HF-cached birefnet.py: torch.linspace runs on meta device due to
# PyTorch's device context manager; force CPU to avoid .item() crash.
# (File is downloaded on first run; supervisor auto-restart applies the fix)
for f in /workspace/.hf_home/modules/transformers_modules/briaai/RMBG*/*/birefnet.py; do
    [ -f "$f" ] && sed -i "s/torch.linspace(0, drop_path_rate, sum(depths))/torch.linspace(0, drop_path_rate, sum(depths), device='cpu')/g" "$f"
done 2>/dev/null || true

cd /workspace/trellis2-webui
exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
LAUNCHER_EOF
chmod +x "$WEBUI_LAUNCHER"

# Create Supervisor config
cat > "$SUPERVISOR_CONF" << EOF
[program:trellis-webui]
environment=PROC_NAME="%(program_name)s"
command=${WEBUI_LAUNCHER}
directory=/workspace/trellis2-webui
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
startsecs=10
stopwaitsecs=30
EOF

# Add to Instance Portal
PORTAL_YAML="/etc/portal.yaml"
if [ -f "$PORTAL_YAML" ] && ! grep -q "TRELLIS.2 WebUI" "$PORTAL_YAML"; then
    python3 -c "
import yaml, sys
portal = '$PORTAL_YAML'
with open(portal) as f:
    data = yaml.safe_load(f) or []
entry = {
    'name': 'TRELLIS.2 WebUI',
    'listen_port': ${PORT},
    'proxy_port': 1${PORT},
    'metrics_port': 0,
    'custom_proxy_url': '',
    'proxy_active': True,
    'path': '/'
}
if isinstance(data, list):
    data.append(entry)
elif isinstance(data, dict):
    data['trellis-webui'] = entry
else:
    data = [entry]
with open(portal, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
print('  Added TRELLIS.2 WebUI to portal.yaml')
"
    # Restart Caddy to pick up the new portal entry
    supervisorctl restart caddy 2>/dev/null || true
fi

# Load the new service
supervisorctl reread
supervisorctl update

echo ""
echo "========================================"
echo "  TRELLIS.2 WebUI registered as Supervisor service"
echo "  Port: ${PORT}"
echo "  Status: supervisorctl status trellis-webui"
echo "  Logs:   supervisorctl tail -f trellis-webui"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
echo ""
