# trellis2-webui

Web UI for [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — image-to-3D generation.

This is a standalone web application that uses TRELLIS.2 as a dependency. It provides a browser-based interface for generating 3D assets from images with full PBR materials.

## Features

- **Single Image → 3D** — upload one image, get a 3D model
- **Multi-View → 3D** — upload 2-6 views for higher accuracy
- **Re-Texturing** — apply new textures to existing meshes
- **Interactive Viewer** — preview renders + Three.js 3D viewer
- **GLB Export** — download production-ready GLB files
- **Quality Presets** — fast / balanced / high / max with surface-type tuning

## Architecture

```
trellis2-webui/          ← This repo
├── app.py               ← FastAPI backend (uses trellis2 as a library)
├── webui/               ← Vite frontend (HTML/JS/CSS)
├── assets/hdri/         ← HDRI environment maps for rendering
├── vastai_setup.sh      ← Vast.ai provisioning script
└── pyproject.toml

TRELLIS.2/               ← Upstream repo (installed separately)
├── trellis2/            ← Python package (pipelines, models, etc.)
├── o-voxel/             ← CUDA extension (submodule)
└── pyproject.toml       ← Declares all CUDA extension deps
```

The WebUI imports from `trellis2` (pipelines, renderers, utils) and `o_voxel` (mesh post-processing). These are installed from the upstream TRELLIS.2 repo via a two-phase install process.

## Quick Start (Vast.ai)

See [vastai_template.md](vastai_template.md) for full deployment instructions.

**TL;DR:** Create a template with `vastai/base-image:cuda-13.2.0-auto` and set `vastai_setup.sh` as the on-start script.

## RTX 4080 / 16 GB VRAM

The WebUI now has four execution profiles. On GPUs below 24 GB, startup recommends **16GB Fast** and enables CPU offload automatically. The app still starts below 12 GB; generation is marked experimental.

| Profile | Geometry | Sampling | Offload in auto mode | Guidance |
|---|---:|---:|---|---|
| **16GB Fast** | 512 | 12 steps | On | Conservative starting point for 12–23 GB GPUs |
| **16GB Quality** | 512 | 20 steps | On | More sampling at the same geometry resolution |
| **24GB+ High** | 1024 cascade | 30 steps | Off at 24 GB+ | High-memory path; 1024 may still need more memory |
| **Maximum** | 1536 cascade | 50 steps | Off at 24 GB+ | Experimental; more than 16 GB may be required |

**No 16 GB inference benchmark has been completed in this checkout.** The 512 profiles are conservative settings, not a claim that the RTX 4080 fit is already measured. The benchmark matrix below must be run on the target installation before describing a mode as validated. In particular, treat 1024/1536 and 4096 texture baking as experimental on 16 GB.

### Execution and memory behavior

- TRELLIS.2 components are loaded from their checkpoint files into CPU memory at startup. The WebUI does not move the full pipeline to CUDA for 16 GB profiles.
- For low-memory runs, the image conditioner processes each reference image one at a time. Its tokens are copied to CPU and concatenated there, so all 2–6 views remain part of the condition without holding a six-image encoder batch on the GPU. The final condition and downstream cross-attention still grow with view count; serial encoding reduces conditioner peak memory but does not guarantee every 6-view workload fits.
- View labels (Front, Left, Top, etc.) are organizational metadata. The released TRELLIS.2 checkpoint consumes image tokens, not camera labels.
- The fork-side multi-view adaptation concatenates each view's DINO patch tokens into the cross-attention sequence. It retains all images but is not an upstream camera-aware interface; output quality and scaling for 2–6 views still require inference validation.
- Sparse structure, shape, material, decoder, preview, and export work are separate stages. Upstream's low-VRAM model calls move their active network to CUDA and return it to CPU; the WebUI releases stage intermediates and clears the CUDA cache at stage boundaries.
- The re-texturing pipeline loads lazily and reuses the image pipeline's texture flow and decoder weights. This avoids loading a second copy of those weights into CPU RAM.
- The official 4B pipeline config names BF16 flow checkpoints and FP16 decoders. The app preserves checkpoint precision; it does not cast all tensors to a different dtype. The UI reports CUDA BF16 capability.
- Attention defaults to **Auto → PyTorch SDPA**. The fork's sparse-attention adapter enables SDPA for TRELLIS.2's variable-length sparse attention. `flash_attn`, `flash_attn_3`, and `xformers` can be explicitly selected only when installed and supported by that build.
- Initial RAM guidance: **32 GB system RAM minimum; 48 GB or more recommended** for CPU-backed model weights and WSL/Linux overhead. This is a planning estimate, not a measured minimum.
- Generation logs `[VRAM]` allocated, reserved, and peak allocated memory at stage boundaries. The response also includes a stage list and peak allocated VRAM; the WebUI displays the peak after generation. CUDA OOMs return the failing stage and a lower-profile suggestion. GLB/OBJ export has one explicit no-remesh fallback.

The following is the intended/code-inspected residency for low-VRAM jobs, not a measured peak profile:

| Stage | Components and large tensors on CUDA |
|---|---|
| Preprocess / condition | `rembg_model` only while processing an opaque image; `image_cond_model` (DINOv3) only during each sequential view-encoding pass. Encoded views stay on CPU. |
| Sparse structure | `sparse_structure_flow_model`, then `sparse_structure_decoder`; 512 condition and sparse intermediates remain as required by the upstream calls. |
| Shape sampling | The active `shape_slat_flow_model_512` or `_1024`. Cascade calls the low-resolution flow, `shape_slat_decoder` upsampler, and high-resolution flow in sequence; both condition resolutions are needed for the cascade. |
| Material sampling | The selected `tex_slat_flow_model_512` or `_1024` plus shape SLat and its conditioning. Shape SLat is required by TRELLIS.2's `concat_cond` path; the sampled texture SLat is moved to CPU before geometry decode. |
| Decode / preview | `shape_slat_decoder` then `tex_slat_decoder`; texture SLat is copied back for texture decode. Meshes/voxel attributes and preview renderer work remain on CUDA. |
| GLB/OBJ export | Inference networks stay on CPU; O-Voxel/CuMesh, vertex data, PBR volume, UV rasterization, and texture-bake scratch buffers use CUDA. |

High-memory generation profiles preserve the resident image-to-3D path: all models in that pipeline are moved to CUDA for the job. Exact peak stage and how much 2–6 views change downstream VRAM are **unknown until inference is measured**. Code inspection says sequential conditioning avoids a multi-image DINO batch, while the final cross-attention condition grows with the number of views. Sparse tensor caches are cleared when the texture latent is staged to CPU; other intermediates are released at their last stage use.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `TRELLIS_EXECUTION_PRESET` | `auto` | `auto`, `16gb-fast`, `16gb-quality`, `24gb-high`, or `maximum`; the WebUI can change the profile per job |
| `TRELLIS_CPU_OFFLOAD` | `auto` | `auto`, `1`, or `0`; auto offloads below 24 GB and uses full residency for high-memory profiles |
| `TRELLIS_ATTENTION_BACKEND` | `auto` | `auto` selects SDPA; explicit options are `sdpa`, `flash_attn`, `flash_attn_3`, and `xformers` |
| `TRELLIS_MODEL_PATH` | `/workspace/models/TRELLIS.2-4B` | Local checkpoint directory or Hugging Face model ID |
| `TRELLIS_PORT` | `8000` | WebUI port |

The resolution control can override a profile. On GPUs below 24 GB, selecting 1024 or 1536 shows an experimental warning; the backend logs the selected resolution and returns a useful error if CUDA runs out of memory. Camera labels never alter model conditioning.

### Benchmark and validation

Start the WebUI, then run the benchmark client from another terminal. Supply one path per view; multi-view inputs must be exactly 2, 4, or 6 images.

```bash
python benchmark_vram.py --views 1 --preset 16gb-fast --image /path/to/front.png
python benchmark_vram.py --views 2 --preset 16gb-fast --image /path/to/front.png --image /path/to/left.png
python benchmark_vram.py --views 4 --preset 16gb-fast --image /path/to/front.png --image /path/to/left.png --image /path/to/back.png --image /path/to/right.png
python benchmark_vram.py --views 6 --preset 16gb-fast --image /path/to/front.png --image /path/to/front-left.png --image /path/to/left.png --image /path/to/back.png --image /path/to/right.png --image /path/to/top.png
python benchmark_vram.py --views 1 --preset 16gb-quality --image /path/to/front.png
```

The script calls generation and GLB export, writes the GLB under `benchmark-output/`, and prints success, selected resolution, view count, generation seconds, stage VRAM, export peak, and GLB size. Run it separately for 2, 4, and 6 views using the 16GB profile. Record results in the table before making fit claims:

| Case | Result | Peak generation VRAM | Time | Resolution | Views | GLB size |
|---|---|---:|---:|---:|---:|---:|
| A — 16GB Fast | Not run: model weights and PyTorch/CUDA environment were unavailable in the inspected host | — | — | 512 | 1 | — |
| B — 16GB Fast | Not run | — | — | 512 | 2 | — |
| C — 16GB Fast | Not run | — | — | 512 | 4 | — |
| D — 16GB Fast | Not run | — | — | 512 | 6 | — |
| E — highest fitting mode | Not run | — | — | — | 1 | — |

**Current inspected host:** Windows reports an RTX 4080 and NVIDIA driver 591.86; WSL2 can see the GPU, but WSL had no PyTorch, CUDA toolkit, TRELLIS.2 weights, or DINOv3 weights, and reported about 15 GiB available RAM plus 4 GiB swap at inspection time. Therefore this checkout has no measured inference VRAM, generation time, GLB output size, or tested PyTorch/CUDA combination. The local setup script requests PyTorch 2.10.0 with CUDA 13.0; upstream's README describes PyTorch 2.6.0/CUDA 12.4 as its default path. Neither was run here.

### Upstream integration and output compatibility

The fork's local setup pins TRELLIS.2 to commit `5565d240c4a494caaf9ece7a554542b76ffa36d3`. Neither that pinned pipeline nor inspected upstream `main` defines `run_multiview`; the old WebUI endpoint called that missing method. The fork now builds the conditioning sequence in the WebUI backend and uses the upstream `sample_sparse_structure`, `sample_shape_slat[_cascade]`, `sample_tex_slat`, and decoder APIs.

Setup continues to apply the fork's isolated compatibility patches: `patches/sparse_attn_sdpa.py` adds the PyTorch SDPA sparse-attention branch, `patches/dinov3_layer_compat.py` handles Transformers' DINOv3 layer layout, `patches/decode_latent_oom_fix.py` reduces retained decoder intermediates in upstream's original `run()` path, and `patches/normal_map_bake.py` adds the existing tangent-space normal-map bake to O-Voxel. The setup scripts also make background-remover loading optional when BiRefNet dependencies are unavailable. This low-VRAM generation path calls upstream stages directly, so the `decode_latent_oom_fix.py` change does not replace or wrap the new staged runner.

GLB export still uses the upstream O-Voxel `to_glb` path with the same `[-0.5, 0.5]` AABB, grid resolution, PBR attribute layout, UV baking, and coordinate conversion. The fork's normal-map patch remains applied. It does not change mesh scale, coordinate system, UV orientation, or GLB material packing, so Blender 4.x and Godot 4.x remain the intended consumers; real import validation still needs to be performed on generated files.

## Local Development

### Prerequisites

- NVIDIA GPU; 16 GB profiles are conservative but not yet inference-validated, while 24 GB+ is recommended for high-resolution modes
- CUDA Toolkit 12.4+ or 13.x
- Python 3.10+
- Node.js 18+

### Setup

```bash
# 1. Clone & build TRELLIS.2 (with CUDA extensions)
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2
uv venv && uv sync --no-build-isolation

# 2. Clone this repo and install into the same venv
cd ..
git clone https://github.com/YOUR_USER/trellis2-webui.git
cd trellis2-webui
source ../TRELLIS.2/.venv/bin/activate
pip install -e .

# 3. Download the model
huggingface-cli download microsoft/TRELLIS.2-4B --local-dir ~/models/TRELLIS.2-4B

# 4. Build frontend
cd webui && npm install && npm run build && cd ..

# 5. Run
TRELLIS_MODEL_PATH=~/models/TRELLIS.2-4B python app.py
```

Open http://localhost:8000

### Frontend Development

For hot-reload during UI development:

```bash
# Terminal 1: backend
TRELLIS_MODEL_PATH=~/models/TRELLIS.2-4B python app.py

# Terminal 2: frontend dev server (proxies /api to backend)
cd webui && npm run dev
```

Open http://localhost:5173

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRELLIS_MODEL_PATH` | `/workspace/models/TRELLIS.2-4B` | Model directory |
| `TRELLIS_PORT` | `8000` | Server port |
| `ATTN_BACKEND` | `sdpa` fallback | Legacy fallback; prefer `TRELLIS_ATTENTION_BACKEND` |
| `SKIP_MODEL_DOWNLOAD` | `0` | Set to `1` to skip HuggingFace model downloads during setup |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for gated/private model access |
| `TRELLIS_EXECUTION_PRESET` | `auto` | Startup/default execution profile; see the RTX 4080 section |
| `TRELLIS_CPU_OFFLOAD` | `auto` | Stage model residency policy: `auto`, `1`, or `0` |
| `TRELLIS_ATTENTION_BACKEND` | `auto` | Attention backend; `auto` selects SDPA |

## Manual Model Provisioning

Some servers have unreliable HuggingFace connectivity. Set `SKIP_MODEL_DOWNLOAD=1` in your Vast.ai template env vars to skip automatic downloads and provide models manually.

The setup script expects models at these paths:

| Model | Path | Size |
|---|---|---|
| TRELLIS.2-4B | `/workspace/models/TRELLIS.2-4B/` | ~16 GB |
| DINOv3 ViT-L/16 | `/workspace/models/dinov3-vitl16-pretrain-lvd1689m/` | ~1.2 GB |

**Transfer models via rsync:**

```bash
# From a machine that has the models downloaded
rsync -avP TRELLIS.2-4B/ root@<vast-host>:/workspace/models/TRELLIS.2-4B/
rsync -avP dinov3-vitl16-pretrain-lvd1689m/ root@<vast-host>:/workspace/models/dinov3-vitl16-pretrain-lvd1689m/
```

Once models are in place, the DINOv3 pipeline config is patched automatically on the next run of `vastai_setup.sh`. You can also restart the WebUI service directly:

```bash
supervisorctl restart trellis-webui
```

## License

MIT — see [LICENSE](LICENSE).

TRELLIS.2 and some dependencies (nvdiffrast, nvdiffrec) have their own licenses.
