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

The WebUI now has four execution profiles. On GPUs below 24 GB, startup recommends **16GB Fast** and uses staged model loading with CPU-backed conditioning. The app still starts below 12 GB; generation is marked experimental.

| Profile | Geometry | Sampling | Offload in auto mode | Guidance |
|---|---:|---:|---|---|
| **16GB Fast** | 512 | 12 steps | On | Conservative starting point for 12–23 GB GPUs |
| **16GB Quality** | 512 | 20 steps | On | Single-view fit measured; multi-view Quality is untested |
| **24GB+ High** | 1024 cascade | 30 steps | Off at 24 GB+ | High-memory path; 1024 may still need more memory |
| **Maximum** | 1536 cascade | 50 steps | Off at 24 GB+ | Experimental; more than 16 GB may be required |

**Measured on an RTX 4080 16 GB under WSL2:** 16GB Fast completed at 512 with 1, 2, 4, and 6 reference images, including GLB export. 16GB Quality completed at 512 with one image. These runs verify completion and allocator peaks for the test inputs; they do not establish same-object multi-view fidelity or guarantee identical memory behavior with other drivers, inputs, or concurrent GPU workloads. 1024/1536 generation is not loaded on GPUs below 24 GB. The 24GB+ and Maximum profiles, and 4096 texture baking, remain untested here.

### Execution and memory behavior

- In the 16 GB path, DINOv3 stays backed by CPU memory between view encodes and generation components are loaded from checkpoints only for their active stage. On 24 GB+ devices, upstream's full image-to-3D model set remains available for the high-memory path.
- For low-memory runs, the image conditioner processes each reference image one at a time. Its tokens are copied to CPU and concatenated there, so all 2–6 views remain part of the condition without holding a six-image encoder batch on the GPU. The final condition and downstream cross-attention still grow with view count; serial encoding reduces conditioner peak memory but does not guarantee every 6-view workload fits.
- View labels (Front, Left, Top, etc.) are organizational metadata. The released TRELLIS.2 checkpoint consumes image tokens, not camera labels.
- The fork-side multi-view adaptation concatenates each view's DINO patch tokens into the cross-attention sequence. It retains all images but is not an upstream camera-aware interface; output quality and scaling for 2–6 views still require inference validation.
- On GPUs below 24 GB, the WebUI starts the upstream pipeline without preloading its full set of shape/material components into host RAM. It loads the needed 512 component from the checkpoint for each stage, then drops that component, collects it, and clears the CUDA cache after its last use. The image conditioner remains available on CPU and is moved to CUDA only while encoding each view. This reload-from-checkpoint policy uses disk I/O to avoid retaining unused 1024 flow weights in CPU RAM; high-memory startup preserves upstream's full model set.
- Sparse structure, shape, material, decoder, preview, and export work are separate stages. Stage tensors are released when their last downstream consumer finishes. The sampled material latent is copied to CPU while shape decoding runs, then copied back for texture decoding.
- The re-texturing pipeline loads lazily and reuses the image pipeline's texture flow and decoder weights. This avoids loading a second copy of those weights into CPU RAM.
- The official 4B pipeline config names BF16 flow checkpoints and FP16 decoders. The app preserves checkpoint precision; it does not cast all tensors to a different dtype. The UI reports CUDA BF16 capability.
- Attention defaults to **Auto**: it uses Flash Attention when that extension imports successfully, otherwise the fork's compatibility patch enables PyTorch SDPA. This WSL installation had no importable `flash_attn` module, so Auto selected PyTorch SDPA for the measured matrix. `sdpa`, `flash_attn`, `flash_attn_3`, and `xformers` can be explicitly selected; Flash Attention 3 and xFormers require a compatible installed build.
- Test host: **32 GiB Windows RAM**, WSL capped at **16 GiB RAM with 8 GiB swap**. The 1/2/4/6-view Fast runs and 20-step single-view Quality run completed with this configuration. An earlier all-model CPU preload was killed by WSL's system OOM at about 15.5 GiB process RSS; the current lazy-load path avoids preloading the unused 1024 flow networks. This is a tested setup, not a minimum-RAM guarantee.
- Generation logs `[VRAM]` allocated, reserved, and peak allocated memory at stage boundaries. The response also includes a stage list and peak allocated VRAM; the WebUI displays the peak after generation. CUDA OOMs return the failing stage and a lower-profile suggestion. GLB/OBJ export has one explicit no-remesh fallback.
- Automatic removal via `briaai/RMBG-2.0` is gated. If access is unavailable, the WebUI logs a non-fatal warning and continues; use references with transparent masks or clean backgrounds when automatic background removal is unavailable.

The following describes the low-VRAM residency path. The peak table below records measured PyTorch allocator peaks for the test machine:

| Stage | Components and large tensors on CUDA |
|---|---|
| Preprocess / condition | `rembg_model` only while processing an opaque image; `image_cond_model` (DINOv3) only during each sequential view-encoding pass. Encoded views stay on CPU. |
| Sparse structure | `sparse_structure_flow_model`, then `sparse_structure_decoder`; 512 condition and sparse intermediates remain as required by the upstream calls. |
| Shape sampling | The active `shape_slat_flow_model_512` or `_1024`. Cascade calls the low-resolution flow, `shape_slat_decoder` upsampler, and high-resolution flow in sequence; both condition resolutions are needed for the cascade. |
| Material sampling | The selected `tex_slat_flow_model_512` or `_1024` plus shape SLat and its conditioning. Shape SLat is required by TRELLIS.2's `concat_cond` path; the sampled texture SLat is moved to CPU before geometry decode. |
| Decode / preview | `shape_slat_decoder` then `tex_slat_decoder`; texture SLat is copied back for texture decode. Meshes/voxel attributes and preview renderer work remain on CUDA. |
| GLB/OBJ export | Inference networks stay on CPU; O-Voxel/CuMesh, vertex data, PBR volume, UV rasterization, and texture-bake scratch buffers use CUDA. |

High-memory generation profiles preserve the resident image-to-3D path: all models in that pipeline are moved to CUDA for the job. They were not tested on this 16 GB machine. In the measured 512 runs, sequential conditioning kept the conditioner peak at 1.37 GiB for each tested view count; final cross-attention still consumes all view tokens, and the sparse/shape/material stages peaked at 2.62–2.89 GiB allocated. Sparse tensor caches are cleared when the texture latent is staged to CPU; other intermediates are released at their last stage use.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `TRELLIS_EXECUTION_PRESET` | `auto` | `auto`, `16gb-fast`, `16gb-quality`, `24gb-high`, or `maximum`; the WebUI can change the profile per job |
| `TRELLIS_CPU_OFFLOAD` | `auto` | `auto`, `1`, or `0`; auto offloads below 24 GB and uses full residency for high-memory profiles |
| `TRELLIS_ATTENTION_BACKEND` | `auto` | `auto` selects SDPA; explicit options are `sdpa`, `flash_attn`, `flash_attn_3`, and `xformers` |
| `TRELLIS_MODEL_PATH` | `/workspace/models/TRELLIS.2-4B` | Local checkpoint directory or Hugging Face model ID |
| `TRELLIS_PORT` | `8000` | WebUI port |

The resolution control can override a profile. On GPUs below 24 GB, selecting 1024 or 1536 shows an experimental warning; the backend logs the selected resolution and returns a useful error if CUDA runs out of memory. Camera labels never alter model conditioning.

### Start the validated WSL installation

In WSL Bash, from the installed checkout:

```bash
cd "$HOME/trellis2-runtime/trellis2-webui"
export TRELLIS_DIR="$HOME/trellis2-runtime/TRELLIS.2"
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export LIBRARY_PATH="/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-13.0/bin:$HOME/.local/bin:$HOME/.nvm/versions/node/v22.19.0/bin:$PATH"
export TRELLIS_EXECUTION_PRESET=auto TRELLIS_CPU_OFFLOAD=auto
export TRELLIS_ATTENTION_BACKEND=auto ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
./run_local.sh
```

Open `http://127.0.0.1:8000/`. The startup detects the RTX 4080, selects 16GB Fast, and Auto uses SDPA on this environment because Flash Attention is unavailable.

### Benchmark and validation

Start the WebUI, then run the benchmark client from another terminal. Supply one path per view; multi-view inputs must be exactly 2, 4, or 6 images.

```bash
python benchmark_vram.py --views 1 --preset 16gb-fast --image /path/to/front.png
python benchmark_vram.py --views 2 --preset 16gb-fast --image /path/to/front.png --image /path/to/left.png
python benchmark_vram.py --views 4 --preset 16gb-fast --image /path/to/front.png --image /path/to/left.png --image /path/to/back.png --image /path/to/right.png
python benchmark_vram.py --views 6 --preset 16gb-fast --image /path/to/front.png --image /path/to/front-left.png --image /path/to/left.png --image /path/to/back.png --image /path/to/right.png --image /path/to/top.png
python benchmark_vram.py --views 1 --preset 16gb-quality --image /path/to/front.png
```

The script calls generation and GLB export, writes the GLB under `benchmark-output/`, and prints success, selected resolution, view count, generation seconds, stage VRAM, export peak, and GLB size. The following are actual runs on 2026-09-27. Peak memory is PyTorch `max_memory_allocated` for generation; export peak is the corresponding measured allocator peak from export. Times cover generation and export separately. The bundled input images are distinct examples, not multiple photographs of one object, so the multi-view cases validate the execution path and resource use rather than view-consistency quality.

| Case | Result | Peak generation / export VRAM | Generation / export time | Resolution | Views | GLB size |
|---|---|---:|---:|---:|---:|---:|
| A — 16GB Fast | Success | 2.62 / 1.28 GiB | 56.599 / 21.742 s | 512 | 1 | 38,570,260 B |
| B — 16GB Fast | Success | 2.82 / 1.85 GiB | 54.095 / 23.645 s | 512 | 2 | 43,151,208 B |
| C — 16GB Fast | Success | 2.87 / 1.87 GiB | 57.621 / 24.878 s | 512 | 4 | 39,712,584 B |
| D — 16GB Fast | Success | 2.71 / 1.67 GiB | 63.085 / 22.054 s | 512 | 6 | 38,543,632 B |
| E — 16GB Quality | Success, single-view only | 2.81 / 1.26 GiB | 55.901 / 22.280 s | 512 | 1 | 49,826,512 B |

Auto resolved to PyTorch SDPA for all cases because the `flash_attn` package was not installed in this environment. Test A also ran with explicit `TRELLIS_ATTENTION_BACKEND=sdpa`, confirming the backend selection rather than relying only on the runtime label. Multi-view SDPA completed all 2/4/6-view cases.

**Test environment:** WSL2 Ubuntu 24.04.3; Python 3.12; PyTorch 2.10.0+cu130; CUDA runtime/toolkit 13.0; Windows NVIDIA driver 591.86; RTX 4080 with 16,376 MiB reported by CUDA; BF16 supported; PyTorch SDPA attention. WSL was configured for 16 GiB RAM and 8 GiB swap on a Windows host with 32 GiB RAM. At the post-test snapshot, WSL reported 10 GiB available and 2.1 GiB swap in use. Tests were run with the local TRELLIS.2-4B and DINOv3 weights. Windows-native inference was not tested.

The low-memory re-texturing endpoint also completed at 512: 3.205 GiB peak allocated, returning a 28,462,672 B GLB. `trimesh` parsed all five benchmark GLBs as one mesh with a `PBRMaterial`. Godot 4.7.2 headless imported and instantiated a generated GLB successfully. Blender 4.x was not installed on the test host, so its import remains unverified.

The measured highest tested profile on this GPU is 16GB Quality at 512 with 20 sampling steps for one view. 16GB Fast was measured at 1/2/4/6 views; 16GB Quality was not tested with multiple views. The 1024/1536 options are intentionally unavailable below 24 GB in the low-memory runtime rather than being presented as 16 GB-safe. Godot 4.7.2 import was exercised; Blender 4.x import was unavailable for validation. The GLB exporter continues to use the existing upstream coordinate, scale, UV, and PBR conventions.

### Upstream integration and output compatibility

The fork's local setup pins TRELLIS.2 to commit `5565d240c4a494caaf9ece7a554542b76ffa36d3`. Neither that pinned pipeline nor inspected upstream `main` defines `run_multiview`; the old WebUI endpoint called that missing method. The fork now builds the conditioning sequence in the WebUI backend and uses the upstream `sample_sparse_structure`, `sample_shape_slat[_cascade]`, `sample_tex_slat`, and decoder APIs.

Setup continues to apply the fork's isolated compatibility patches: `patches/sparse_attn_sdpa.py` adds the PyTorch SDPA sparse-attention branch and adds `sdpa` to upstream's backend allowlist, `patches/dinov3_layer_compat.py` handles Transformers' DINOv3 layer layout, `patches/decode_latent_oom_fix.py` reduces retained decoder intermediates in upstream's original `run()` path, and `patches/normal_map_bake.py` adds the existing tangent-space normal-map bake to O-Voxel. The setup scripts also make background-remover loading optional when BiRefNet dependencies are unavailable. This low-VRAM generation path calls upstream stages directly, so the `decode_latent_oom_fix.py` change does not replace or wrap the new staged runner.

GLB export still uses the upstream O-Voxel `to_glb` path with the same `[-0.5, 0.5]` AABB, grid resolution, PBR attribute layout, UV baking, and coordinate conversion. The fork's normal-map patch remains applied. It does not change mesh scale, coordinate system, UV orientation, or GLB material packing. Godot 4.7.2 import was checked; Blender 4.x import should be checked on a machine with Blender installed.

## Local Development

### Prerequisites

- NVIDIA GPU; 16GB Fast at 512 is inference-validated on WSL2/RTX 4080 for 1/2/4/6 views, while 24 GB+ is required to load high-resolution components in this fork
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
