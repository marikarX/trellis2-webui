import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Auto prefers installed Flash Attention for variable-length sparse sequences;
# the padded PyTorch SDPA adapter is the dependency-free fallback.
_requested_attention = os.environ.get(
    "TRELLIS_ATTENTION_BACKEND", os.environ.get("ATTN_BACKEND", "auto")
).strip().lower()
REQUESTED_ATTENTION_BACKEND = _requested_attention
if _requested_attention == "auto":
    try:
        __import__("flash_attn")
        _requested_attention = "flash_attn"
    except Exception:
        _requested_attention = "sdpa"
if _requested_attention not in {"sdpa", "flash_attn", "flash_attn_3", "xformers"}:
    raise RuntimeError("TRELLIS_ATTENTION_BACKEND must be auto, sdpa, flash_attn, flash_attn_3, or xformers")
ATTENTION_BACKEND = _requested_attention
os.environ["ATTN_BACKEND"] = ATTENTION_BACKEND
os.environ["SPARSE_ATTN_BACKEND"] = ATTENTION_BACKEND

import asyncio
import gc
import uuid
import shutil
import json
import io
import base64
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse import config as sparse_config
sparse_config.ATTN = ATTENTION_BACKEND
from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
from trellis2.pipelines import samplers as trellis_samplers
from trellis2 import models as trellis_models
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel
from runtime_config import (
    PROFILES,
    VramMeter,
    get_gpu_info,
    is_cuda_oom,
    oom_detail,
    recommend_profile,
    resolve_cpu_offload,
    resolve_profile,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
TMP_DIR = APP_DIR / "tmp_webui"
WEBUI_DIR = APP_DIR / "webui" / "dist"
ASSETS_DIR = APP_DIR / "assets"
MAX_SEED = 100000  # Cap random seeds at 100k for all stages

# Model path: env var > default location for vast.ai
MODEL_PATH = os.environ.get(
    'TRELLIS_MODEL_PATH',
    '/workspace/models/TRELLIS.2-4B'
)

# Use second GPU for mesh processing if available, otherwise same GPU
GENERATION_DEVICE = "cuda:0"
MESH_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
GPU_INFO = get_gpu_info(torch)
RECOMMENDED_PROFILE = recommend_profile(GPU_INFO.get("total_vram_gb"))
DEFAULT_PROFILE_REQUEST = os.environ.get("TRELLIS_EXECUTION_PRESET", "auto")
CPU_OFFLOAD_REQUEST = os.environ.get("TRELLIS_CPU_OFFLOAD", "auto")

# Upstream loads both 512 and 1024 shape/texture flow networks into host RAM.
# The 16 GB path never calls the 1024 models, and loading those unused 1.3B
# networks consumed the WSL RAM/swap headroom needed to decode a 512 result.
# Keep the complete upstream set on 24 GB+ devices; low-memory devices load only
# components required by the supported 512 path.
_ALL_IMAGE_MODEL_NAMES = tuple(Trellis2ImageTo3DPipeline.model_names_to_load)
_LOW_MEMORY_UNUSED_MODELS = frozenset({
    "shape_slat_flow_model_1024",
    "tex_slat_flow_model_1024",
})
_LOAD_HIGH_RESOLUTION_MODELS = (
    GPU_INFO.get("total_vram_gb") is not None
    and GPU_INFO["total_vram_gb"] >= 24
)
_LOW_MEMORY_IMAGE_MODEL_NAMES = tuple(
    name for name in _ALL_IMAGE_MODEL_NAMES if name not in _LOW_MEMORY_UNUSED_MODELS
)

# Global pipeline references (initialized on startup)
pipeline: Trellis2ImageTo3DPipeline = None  # type: ignore
tex_pipeline: Trellis2TexturingPipeline = None  # type: ignore
envmap: dict = {}

# In-memory state store for CPU numpy arrays (session_id -> dict)
latent_store: dict = {}

# Progress tracking
progress_store: dict = {}  # task_id -> {stage, step, total, done}

# GPU lock — only one generation at a time
gpu_lock = asyncio.Lock()


def force_offload_all():
    """Release model and environment-map allocations at a stage boundary."""
    cleanup_errors = []
    for p in (pipeline, tex_pipeline):
        if p is None:
            continue
        for name, model in getattr(p, "models", {}).items():
            try:
                model.to("cpu")
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        for attr_name in ("image_cond_model", "rembg_model"):
            component = getattr(p, attr_name, None)
            if component is not None:
                try:
                    if hasattr(component, "to"):
                        component.to("cpu")
                    elif hasattr(component, "cpu"):
                        component.cpu()
                except Exception as exc:
                    cleanup_errors.append(f"{attr_name}: {exc}")
        p._device = "cpu"

    for v in envmap.values():
        try:
            if hasattr(v, 'image') and hasattr(v.image, 'to'):
                v.image = v.image.to('cpu')
            if hasattr(v, '_nvdiffrec_envlight'):
                delattr(v, '_nvdiffrec_envlight')
        except Exception as exc:
            cleanup_errors.append(f"environment map: {exc}")

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            cleanup_errors.append(f"CUDA cache: {exc}")
    if cleanup_errors:
        print(f"[VRAM] Offload cleanup warnings: {'; '.join(cleanup_errors)}", flush=True)


def load_pipeline_model(model_name: str):
    """Load one upstream component on demand for the low-memory profile."""
    if model_name in pipeline.models:
        return pipeline.models[model_name]
    if model_name not in _ALL_IMAGE_MODEL_NAMES:
        raise KeyError(f"Unknown TRELLIS.2 model component: {model_name}")
    if not _LOAD_HIGH_RESOLUTION_MODELS and model_name not in _LOW_MEMORY_IMAGE_MODEL_NAMES:
        raise RuntimeError(f"{model_name} is unavailable in the 512-only low-memory profile.")

    relative_path = pipeline._pretrained_args["models"][model_name]
    try:
        model = trellis_models.from_pretrained(str(Path(MODEL_PATH) / relative_path))
    except Exception:
        # Match the upstream loader's fallback for components stored in a
        # separate Hugging Face repository (for example the sparse decoder).
        model = trellis_models.from_pretrained(relative_path)
    model.eval()
    pipeline.models[model_name] = model
    print(f"[MODEL] Loaded stage component: {model_name}", flush=True)
    return model


def release_pipeline_models(model_names):
    """Drop low-memory stage weights instead of retaining a second CPU copy."""
    if _LOAD_HIGH_RESOLUTION_MODELS or pipeline is None:
        return
    removed = [pipeline.models.pop(name) for name in model_names if name in pipeline.models]
    if not removed:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize(GENERATION_DEVICE)
    except Exception:
        pass
    removed.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[MODEL] Released stage component(s): {', '.join(model_names)}", flush=True)


def restore_envmaps():
    """Restores envmaps to the GPU for rendering."""
    for v in envmap.values():
        if hasattr(v, 'image') and hasattr(v.image, 'to'):
            v.image = v.image.to(GENERATION_DEVICE)


def configure_pipeline_for_job(p, cpu_offload: bool) -> None:
    """Set the pipeline's device policy before any stage starts."""
    if p is None:
        return
    p.low_vram = cpu_offload
    p._device = GENERATION_DEVICE
    if not cpu_offload:
        # This is the original high-memory path: all components stay resident.
        p.to(GENERATION_DEVICE)


def load_texturing_pipeline_shared():
    """Load only the texturing encoder and reuse image-pipeline model weights.

    The upstream texturing loader otherwise loads duplicate texture flow and
    decoder weights into CPU RAM. The GPU lock ensures the shared module objects
    are never used by both pipeline wrappers at once.
    """
    config_path = Path(MODEL_PATH) / "texturing_pipeline.json"
    if config_path.exists():
        texturing_config_path = config_path
    else:
        from huggingface_hub import hf_hub_download
        texturing_config_path = Path(hf_hub_download(MODEL_PATH, "texturing_pipeline.json"))
    args = json.loads(texturing_config_path.read_text(encoding="utf-8"))["args"]
    load_pipeline_model("tex_slat_decoder")
    load_pipeline_model("tex_slat_flow_model_512")
    if _LOAD_HIGH_RESOLUTION_MODELS:
        load_pipeline_model("tex_slat_flow_model_1024")

    shape_encoder_path = args["models"]["shape_slat_encoder"]
    shape_encoder = trellis_models.from_pretrained(f"{MODEL_PATH}/{shape_encoder_path}")
    models = {
        "shape_slat_encoder": shape_encoder,
        "tex_slat_decoder": pipeline.models["tex_slat_decoder"],
        "tex_slat_flow_model_512": pipeline.models["tex_slat_flow_model_512"],
    }
    if "tex_slat_flow_model_1024" in pipeline.models:
        models["tex_slat_flow_model_1024"] = pipeline.models["tex_slat_flow_model_1024"]
    sampler_cfg = args["tex_slat_sampler"]
    return Trellis2TexturingPipeline(
        models=models,
        tex_slat_sampler=getattr(trellis_samplers, sampler_cfg["name"])(**sampler_cfg["args"]),
        tex_slat_sampler_params=args["tex_slat_sampler"]["params"],
        shape_slat_normalization=args["shape_slat_normalization"],
        tex_slat_normalization=args["tex_slat_normalization"],
        image_cond_model=pipeline.image_cond_model,
        rembg_model=pipeline.rembg_model,
        low_vram=True,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TRELLIS.2 WebUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pil_image(file: UploadFile) -> Image.Image:
    data = file.file.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")

def get_session_dir(session_id: str) -> Path:
    d = TMP_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def make_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"

def sanitize_model_name(name: str) -> str:
    """Sanitize user-provided model name for safe filesystem and material use."""
    import re
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9_\-]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name[:64] if name else 'model'

def render_preview_images(mesh, envmap_dict) -> dict:
    mesh.simplify(16777216)
    images = render_utils.render_snapshot(
        mesh, resolution=1024, r=2, fov=36, nviews=8, envmap=envmap_dict
    )
    result = {}
    for key, frames in images.items():
        result[key] = []
        for frame in frames:
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            result[key].append(f"data:image/png;base64,{b64}")
    return result

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
def read_environment_map_rgb(path: Path) -> np.ndarray:
    """Read an HDRI with OpenCV, falling back to OpenEXR when needed.

    Some OpenCV wheels are built without OpenEXR support, even when the
    ``OPENCV_IO_ENABLE_OPENEXR`` flag is set. Keep the bundled PBR preview
    lighting available across those builds.
    """
    image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr is not None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    import OpenEXR

    with OpenEXR.File(str(path)) as exr_file:
        channels = exr_file.channels()
        if "RGB" in channels:
            image_rgb = channels["RGB"].pixels
        elif "RGBA" in channels:
            image_rgb = channels["RGBA"].pixels[..., :3]
        elif all(channel in channels for channel in ("R", "G", "B")):
            image_rgb = np.stack([channels[channel].pixels for channel in ("R", "G", "B")], axis=-1)
        else:
            raise ValueError(f"No RGB channels found in environment map: {path}")

    return np.ascontiguousarray(image_rgb, dtype=np.float32)


@app.on_event("startup")
async def startup():
    global pipeline, tex_pipeline, envmap
    TMP_DIR.mkdir(exist_ok=True)

    if _LOAD_HIGH_RESOLUTION_MODELS:
        Trellis2ImageTo3DPipeline.model_names_to_load = list(_ALL_IMAGE_MODEL_NAMES)
        print(f"[MODEL] Loading the full 512/1024 component set for {GPU_INFO['total_vram_gb']:.1f} GB VRAM")
    else:
        Trellis2ImageTo3DPipeline.model_names_to_load = []
        print("[MODEL] Low-memory mode: 512 generation components load one stage at a time")

    print(f"Loading model from: {MODEL_PATH}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.low_vram = True
    if pipeline.rembg_model is None:
        print(
            "[WARN] RMBG-2.0 is unavailable; opaque reference images will be used as-is "
            "without automatic background removal.",
            flush=True,
        )
    tex_pipeline = None  # Construct lazily and share the texture models on first use.
    force_offload_all()

    envmap_data = {}
    hdri_dir = ASSETS_DIR / "hdri"
    for name in ['forest', 'sunset', 'courtyard']:
        exr_path = hdri_dir / f"{name}.exr"
        if exr_path.exists():
            try:
                image_rgb = read_environment_map_rgb(exr_path)
                envmap_data[name] = EnvMap(torch.from_numpy(image_rgb))
            except Exception as exc:
                print(f"[WARN] Could not load preview HDRI {exr_path.name}: {exc}")
    if not envmap_data:
        print("[WARN] No HDRI preview maps loaded; using neutral lighting for the default preview.")
        neutral_envmap = torch.ones((512, 1024, 3), dtype=torch.float32, device="cpu")
        envmap_data["forest"] = EnvMap(neutral_envmap)
    envmap = envmap_data
    force_offload_all()

    print(f"✅ TRELLIS.2 WebUI ready")
    if GPU_INFO.get("cuda_available"):
        print(f"   GPU: {GPU_INFO['gpu_name']} ({GPU_INFO['total_vram_gb']:.1f} GB VRAM)")
        print(f"   Recommended preset: {PROFILES[RECOMMENDED_PROFILE].label}")
        print(f"   CUDA BF16: {'supported' if GPU_INFO['bf16_supported'] else 'not reported'}")
    else:
        print("   CUDA unavailable; generation will report a runtime error if requested.")
    print(f"   Attention backend: {ATTENTION_BACKEND}; checkpoint precision is preserved")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"   Mesh export on: {MESH_DEVICE} ({torch.cuda.get_device_name(MESH_DEVICE)})")


@app.on_event("shutdown")
async def shutdown():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Progress SSE
# ---------------------------------------------------------------------------
@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str):
    async def event_generator():
        while True:
            info = progress_store.get(task_id, {})
            data = json.dumps(info)
            yield f"data: {data}\n\n"
            if info.get("done"):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/runtime")
async def runtime_info():
    """Hardware and safe-default information shown in the WebUI."""
    current_gpu_info = get_gpu_info(torch)
    recommended_profile = recommend_profile(current_gpu_info.get("total_vram_gb"))
    warnings = []
    try:
        profile = resolve_profile(DEFAULT_PROFILE_REQUEST, current_gpu_info)
    except ValueError as exc:
        profile = resolve_profile("auto", current_gpu_info)
        warnings.append(f"Invalid TRELLIS_EXECUTION_PRESET was ignored: {exc}")
    try:
        offload = resolve_cpu_offload(CPU_OFFLOAD_REQUEST, profile, current_gpu_info.get("total_vram_gb"))
    except ValueError as exc:
        offload = True
        warnings.append(f"Invalid TRELLIS_CPU_OFFLOAD was ignored: {exc}")
    if not _LOAD_HIGH_RESOLUTION_MODELS:
        offload = True
        warnings.append("CPU offload is required and enforced by the low-memory model-loading path.")
    total_vram = current_gpu_info.get("total_vram_gb")
    if total_vram is not None and total_vram < 12:
        warnings.append("Below 12 GB VRAM is experimental; the application will still launch.")
    if total_vram is not None and total_vram < 24:
        warnings.append("Low-memory mode is limited to the tested 512 component path; see the README benchmark matrix for RTX 4080 results.")
    if not _LOAD_HIGH_RESOLUTION_MODELS:
        warnings.append("This runtime loads only 512 generation components to reduce host RAM use; 1024/1536 are unavailable.")
    if pipeline is not None and getattr(pipeline, "rembg_model", None) is None:
        warnings.append(
            "RMBG-2.0 is unavailable. Opaque references are accepted as-is without automatic background removal; "
            "transparent images keep their alpha masks."
        )
    attention_labels = {
        "sdpa": "PyTorch SDPA",
        "flash_attn": "Flash Attention",
        "flash_attn_3": "Flash Attention 3",
        "xformers": "xFormers",
    }
    selected_attention = attention_labels.get(ATTENTION_BACKEND, ATTENTION_BACKEND)
    if REQUESTED_ATTENTION_BACKEND == "auto":
        selected_attention = f"Auto ({selected_attention})"
    return {
        **current_gpu_info,
        "recommended_preset": recommended_profile,
        "default_preset": profile.key,
        "default_geometry_resolution": profile.resolution,
        "cpu_offload": offload,
        "attention_backend": selected_attention,
        "precision": "Checkpoint-defined (upstream TRELLIS.2 4B uses BF16 flow weights and FP16 decoders)",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Generation runtime shared by single-image and multi-view routes
# ---------------------------------------------------------------------------
class GenerationOOMError(RuntimeError):
    pass


def _condition_to_device(condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to(GENERATION_DEVICE) for key, value in condition.items()}


def _empty_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_generation(
    input_images: list[Image.Image],
    view_labels: list[str],
    task_id: str,
    seed: int,
    profile,
    resolution: str,
    ss_params: dict[str, Any],
    shape_params: dict[str, Any],
    tex_params: dict[str, Any],
) -> tuple[dict, dict, float | None, list[dict[str, Any]]]:
    """Run explicit TRELLIS.2 stages so their model/tensor lifetimes are visible."""
    global pipeline
    offload = (
        True
        if not _LOAD_HIGH_RESOLUTION_MODELS
        else resolve_cpu_offload(CPU_OFFLOAD_REQUEST, profile, GPU_INFO.get("total_vram_gb"))
    )
    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}.get(resolution)
    if pipeline_type is None:
        raise ValueError("Resolution must be 512, 1024, or 1536.")

    meter = VramMeter(torch, GENERATION_DEVICE)
    stage_open = False
    cond_512_cpu = cond_1024_cpu = None
    cond_512_gpu = cond_1024_gpu = tex_cond_gpu = None
    processed_images = encoded_views = condition_cpu = None
    image_model = features = combined = tex_cond_cpu = None
    coords = shape_slat = tex_slat = None
    meshes = subs = tex_voxels = out_mesh = mesh = None
    shape_mesh = voxel = None
    state = previews = None

    def begin_stage(stage: str, step: int):
        nonlocal stage_open
        meter.begin(stage)
        stage_open = True
        progress_store[task_id] = {"stage": stage, "step": step, "total": 100, "done": False}

    def end_stage():
        nonlocal stage_open
        meter.end()
        stage_open = False

    try:
        begin_stage("Staging inference models", 2)
        force_offload_all()
        configure_pipeline_for_job(pipeline, offload)
        end_stage()
        torch.manual_seed(seed)

        with torch.no_grad():
            begin_stage("Preprocessing reference images", 5)
            processed_images = [pipeline.preprocess_image(image) for image in input_images]
            end_stage()

            # DINOv3 is moved to CUDA once per resolution. Each view is encoded
            # individually and immediately copied to CPU before the next view.
            # Concatenating patch tokens retains all views in the model's existing
            # cross-attention input; this is a fork-side adaptation, not upstream
            # camera-aware multi-view conditioning.
            needed_resolutions = [512] if pipeline_type == "512" else [512, 1024]
            condition_cpu = {}
            for cond_resolution in needed_resolutions:
                begin_stage(f"Conditioning {cond_resolution} ({len(processed_images)} views, sequential)", 14)
                image_model = pipeline.image_cond_model
                if hasattr(image_model, "image_size"):
                    image_model.image_size = cond_resolution
                if offload:
                    image_model.to(GENERATION_DEVICE)
                encoded_views = []
                try:
                    for view_index, image in enumerate(processed_images):
                        features = image_model([image])
                        if features.ndim != 3 or features.shape[0] != 1:
                            raise RuntimeError(f"Expected one feature batch for view {view_index + 1}, got {tuple(features.shape)}")
                        encoded_views.append(features[0].detach().to("cpu"))
                        del features
                    combined = torch.cat(encoded_views, dim=0).unsqueeze(0)
                    condition_cpu[cond_resolution] = {
                        "cond": combined,
                        "neg_cond": torch.zeros_like(combined),
                    }
                    del combined
                finally:
                    if offload:
                        image_model.to("cpu")
                    encoded_views = None
                end_stage()
            cond_512_cpu = condition_cpu.pop(512)
            cond_1024_cpu = condition_cpu.pop(1024) if 1024 in condition_cpu else None
            del condition_cpu, processed_images
            processed_images = None
            _empty_cuda_cache()

            begin_stage("Sparse structure sampling", 26)
            cond_512_gpu = _condition_to_device(cond_512_cpu)
            sparse_resolution = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
            load_pipeline_model("sparse_structure_flow_model")
            load_pipeline_model("sparse_structure_decoder")
            coords = pipeline.sample_sparse_structure(cond_512_gpu, sparse_resolution, 1, ss_params)
            release_pipeline_models(("sparse_structure_flow_model", "sparse_structure_decoder"))
            cond_512_gpu = None
            end_stage()
            if offload:
                _empty_cuda_cache()

            begin_stage("Shape generation", 40)
            shape_model_names = (
                ("shape_slat_flow_model_512",)
                if pipeline_type == "512"
                else ("shape_slat_flow_model_512", "shape_slat_flow_model_1024")
            )
            for model_name in shape_model_names:
                load_pipeline_model(model_name)
            if pipeline_type == "512":
                cond_512_gpu = _condition_to_device(cond_512_cpu)
                shape_slat = pipeline.sample_shape_slat(
                    cond_512_gpu, pipeline.models["shape_slat_flow_model_512"], coords, shape_params
                )
                output_resolution = 512
            elif pipeline_type == "1024":
                cond_1024_gpu = _condition_to_device(cond_1024_cpu)
                shape_slat = pipeline.sample_shape_slat(
                    cond_1024_gpu, pipeline.models["shape_slat_flow_model_1024"], coords, shape_params
                )
                output_resolution = 1024
            else:
                cond_512_gpu = _condition_to_device(cond_512_cpu)
                cond_1024_gpu = _condition_to_device(cond_1024_cpu)
                cascade_resolution = 1024 if pipeline_type == "1024_cascade" else 1536
                shape_slat, output_resolution = pipeline.sample_shape_slat_cascade(
                    cond_512_gpu,
                    cond_1024_gpu,
                    pipeline.models["shape_slat_flow_model_512"],
                    pipeline.models["shape_slat_flow_model_1024"],
                    512,
                    cascade_resolution,
                    coords,
                    shape_params,
                )
            release_pipeline_models(shape_model_names)
            cond_512_gpu = cond_1024_gpu = None
            coords = None
            del coords
            end_stage()
            if offload:
                _empty_cuda_cache()

            begin_stage("Material generation", 61)
            use_512_material_model = pipeline_type == "512"
            tex_cond_cpu = cond_512_cpu if use_512_material_model else cond_1024_cpu
            tex_cond_gpu = _condition_to_device(tex_cond_cpu)
            tex_model_name = "tex_slat_flow_model_512" if use_512_material_model else "tex_slat_flow_model_1024"
            load_pipeline_model(tex_model_name)
            tex_slat = pipeline.sample_tex_slat(
                tex_cond_gpu, pipeline.models[tex_model_name], shape_slat, tex_params
            )
            release_pipeline_models((tex_model_name,))
            tex_cond_gpu = None
            cond_512_cpu = cond_1024_cpu = None
            del cond_512_cpu, cond_1024_cpu
            if offload:
                # Keep the sampled material latent in system RAM while the
                # geometry decoder runs; it is only needed again for texture decode.
                tex_slat.clear_spatial_cache()
                tex_slat = tex_slat.cpu()
            end_stage()
            if offload:
                _empty_cuda_cache()

            # Decode shape first, release its latent, then decode materials.
            # The only duplicated glue below mirrors upstream decode_latent().
            begin_stage("Decoding geometry and PBR voxels", 75)
            load_pipeline_model("shape_slat_decoder")
            meshes, subs = pipeline.decode_shape_slat(shape_slat, output_resolution)
            release_pipeline_models(("shape_slat_decoder",))
            shape_slat = None
            del shape_slat
            if offload:
                _empty_cuda_cache()
                tex_slat = tex_slat.to(GENERATION_DEVICE)
            load_pipeline_model("tex_slat_decoder")
            tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
            release_pipeline_models(("tex_slat_decoder",))
            tex_slat = None
            subs = None
            del tex_slat, subs
            if offload:
                _empty_cuda_cache()
            from trellis2.representations import MeshWithVoxel
            out_mesh = []
            for shape_mesh, voxel in zip(meshes, tex_voxels):
                shape_mesh.fill_holes()
                out_mesh.append(MeshWithVoxel(
                    shape_mesh.vertices,
                    shape_mesh.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / output_resolution,
                    coords=voxel.coords[:, 1:],
                    attrs=voxel.feats,
                    voxel_shape=torch.Size([*voxel.shape, *voxel.spatial_shape]),
                    layout=pipeline.pbr_attr_layout,
                ))
            meshes = tex_voxels = None
            del meshes, tex_voxels
            mesh = out_mesh[0]
            end_stage()
            if offload:
                _empty_cuda_cache()

            begin_stage("Rendering preview", 88)
            restore_envmaps()
            previews = render_preview_images(mesh, envmap)
            end_stage()

            begin_stage("Copying export state to CPU", 96)
            state = {
                "verts": mesh.vertices.detach().cpu().clone(),
                "faces": mesh.faces.detach().cpu().clone(),
                "attrs": mesh.attrs.detach().cpu().clone(),
                "coords": mesh.coords.detach().cpu().clone(),
                "pbr_layout": pipeline.pbr_attr_layout,
                "res": output_resolution,
                "view_labels": list(view_labels),
                "preset": profile.key,
            }
            end_stage()

        generation_peak = meter.peak_gb
        records = list(meter.records)
        progress_store[task_id].update({
            "peak_vram_gb": generation_peak,
            "vram_stages": records,
        })
        return previews, state, generation_peak, records
    except Exception as exc:
        if is_cuda_oom(exc, torch):
            if stage_open:
                try:
                    end_stage()
                except Exception:
                    pass
            stage = meter.stage
            detail = oom_detail(stage, meter.peak_gb, profile, resolution)
            # Drop the original traceback before its tensor-bearing frames can
            # keep failed-stage allocations reachable during cleanup.
            try:
                exc.__traceback__ = None
                exc.__context__ = None
            except Exception:
                pass
            raise GenerationOOMError(detail) from None
        raise
    finally:
        cond_512_cpu = cond_1024_cpu = None
        cond_512_gpu = cond_1024_gpu = tex_cond_gpu = None
        processed_images = encoded_views = condition_cpu = None
        image_model = features = combined = tex_cond_cpu = None
        coords = shape_slat = tex_slat = None
        meshes = subs = tex_voxels = out_mesh = mesh = None
        shape_mesh = voxel = None
        state = previews = None
        try:
            if not _LOAD_HIGH_RESOLUTION_MODELS:
                # On a failed low-memory stage, dropping CUDA/CPU parameters is
                # safer than copying an incomplete model back to host RAM.
                release_pipeline_models(tuple(pipeline.models))
            force_offload_all()
        except Exception as cleanup_error:
            print(f"[VRAM] Cleanup warning: {cleanup_error}", flush=True)
        _empty_cuda_cache()


def _generation_params(
    ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
    shape_guidance_strength, shape_guidance_rescale, shape_sampling_steps, shape_rescale_t,
    tex_guidance_strength, tex_guidance_rescale, tex_sampling_steps, tex_rescale_t,
):
    return (
        {"steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength, "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t},
        {"steps": shape_sampling_steps, "guidance_strength": shape_guidance_strength, "guidance_rescale": shape_guidance_rescale, "rescale_t": shape_rescale_t},
        {"steps": tex_sampling_steps, "guidance_strength": tex_guidance_strength, "guidance_rescale": tex_guidance_rescale, "rescale_t": tex_rescale_t},
    )


def _profile_and_resolution(preset_key: str, requested_resolution: str):
    profile = resolve_profile(preset_key or DEFAULT_PROFILE_REQUEST, GPU_INFO)
    resolution = str(requested_resolution or profile.resolution)
    if resolution not in {"512", "1024", "1536"}:
        raise HTTPException(status_code=422, detail="Resolution must be 512, 1024, or 1536.")
    if resolution != "512" and not _LOAD_HIGH_RESOLUTION_MODELS:
        raise HTTPException(
            status_code=422,
            detail="This low-memory runtime loads only 512 generation components. 1024/1536 require a 24 GB+ GPU preset.",
        )
    warning = profile.warning
    total_vram = GPU_INFO.get("total_vram_gb")
    if total_vram is not None and total_vram < 24 and resolution != "512":
        warning = f"{resolution} resolution is experimental on this {total_vram:.1f} GB GPU and may run out of memory."
    elif resolution != profile.resolution:
        warning = f"Resolution {resolution} overrides the {profile.label} default ({profile.resolution})."
    return profile, resolution, warning


async def _start_generation(images, labels, task_id, seed, profile, resolution, params):
    progress_store[task_id] = {"stage": "Starting", "step": 0, "total": 100, "done": False}
    async with gpu_lock:
        try:
            if latent_store:
                # The UI has no generation history; keep saved GLB files on disk
                # but drop older CPU mesh tensors before starting another job.
                latent_store.clear()
                gc.collect()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_generation(images, labels, task_id, seed, profile, resolution, *params),
            )
            previews, export_state, peak_vram_gb, vram_stages = result
            session_id = str(uuid.uuid4())
            latent_store[session_id] = export_state
            progress_store[task_id] = {
                "stage": "Done", "step": 100, "total": 100, "done": True,
                "peak_vram_gb": peak_vram_gb, "vram_stages": vram_stages,
            }
            return JSONResponse({
                "task_id": task_id,
                "session_id": session_id,
                "seed": seed,
                "previews": previews,
                "num_views": len(images),
                "view_labels": labels,
                "preset": profile.key,
                "resolution": export_state["res"],
                "peak_vram_gb": peak_vram_gb,
                "vram_stages": vram_stages,
                "warning": _profile_and_resolution(profile.key, resolution)[2],
            })
        except GenerationOOMError as exc:
            progress_store[task_id] = {"stage": f"Error: {exc}", "step": 0, "total": 100, "done": True}
            raise HTTPException(status_code=500, detail=str(exc)) from None
        except Exception as exc:
            progress_store[task_id] = {"stage": f"Error: {exc}", "step": 0, "total": 100, "done": True}
            traceback.print_exc()
            try:
                force_offload_all()
            finally:
                _empty_cuda_cache()
            raise HTTPException(status_code=500, detail=str(exc)) from None


# ---------------------------------------------------------------------------
# API: Single Image → 3D
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
    resolution: str = Form(""),
    execution_preset: str = Form(""),
    task_id: str = Form(""),
    ss_guidance_strength: float = Form(7.5),
    ss_guidance_rescale: float = Form(0.7),
    ss_sampling_steps: int = Form(12),
    ss_rescale_t: float = Form(5.0),
    shape_guidance_strength: float = Form(7.5),
    shape_guidance_rescale: float = Form(0.5),
    shape_sampling_steps: int = Form(12),
    shape_rescale_t: float = Form(3.0),
    tex_guidance_strength: float = Form(1.0),
    tex_guidance_rescale: float = Form(0.0),
    tex_sampling_steps: int = Form(12),
    tex_rescale_t: float = Form(3.0),
):
    task_id = task_id.strip() or str(uuid.uuid4())
    if randomize_seed:
        seed = int(np.random.randint(0, MAX_SEED))
    try:
        profile, resolution, _warning = _profile_and_resolution(execution_preset, resolution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    image_pil = load_pil_image(image)
    params = _generation_params(
        ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
        shape_guidance_strength, shape_guidance_rescale, shape_sampling_steps, shape_rescale_t,
        tex_guidance_strength, tex_guidance_rescale, tex_sampling_steps, tex_rescale_t,
    )
    return await _start_generation([image_pil], ["Front"], task_id, seed, profile, resolution, params)


# ---------------------------------------------------------------------------
# API: Multi-View → 3D
# ---------------------------------------------------------------------------
@app.post("/api/generate-multiview")
async def generate_multiview(
    request: Request,
    seed: int = Form(0),
    randomize_seed: bool = Form(True),
    resolution: str = Form(""),
    execution_preset: str = Form(""),
    task_id: str = Form(""),
    ss_guidance_strength: float = Form(7.5),
    ss_guidance_rescale: float = Form(0.7),
    ss_sampling_steps: int = Form(12),
    ss_rescale_t: float = Form(5.0),
    shape_guidance_strength: float = Form(7.5),
    shape_guidance_rescale: float = Form(0.5),
    shape_sampling_steps: int = Form(12),
    shape_rescale_t: float = Form(3.0),
    tex_guidance_strength: float = Form(1.0),
    tex_guidance_rescale: float = Form(0.0),
    tex_sampling_steps: int = Form(12),
    tex_rescale_t: float = Form(3.0),
):
    task_id = task_id.strip() or str(uuid.uuid4())
    if randomize_seed:
        seed = int(np.random.randint(0, MAX_SEED))

    form = await request.form()
    indexed_uploads = []
    labels_by_index: dict[int, str] = {}
    for key, value in form.multi_items():
        if key.startswith("view_label_"):
            try:
                labels_by_index[int(key.rsplit("_", 1)[1])] = str(value)
            except ValueError:
                continue
        elif key.startswith("image_") and hasattr(value, "read") and getattr(value, "filename", None):
            try:
                index = int(key.rsplit("_", 1)[1])
            except ValueError:
                continue
            indexed_uploads.append((index, value))
    indexed_uploads.sort(key=lambda row: row[0])
    if not 2 <= len(indexed_uploads) <= 6:
        raise HTTPException(status_code=422, detail="Multi-view generation requires 2 to 6 images.")

    images, labels = [], []
    for index, upload in indexed_uploads:
        data = await upload.read()
        images.append(Image.open(io.BytesIO(data)).convert("RGBA"))
        label = labels_by_index.get(index, "Custom").strip()
        labels.append(label or "Custom")
    try:
        profile, resolution, _warning = _profile_and_resolution(execution_preset, resolution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    params = _generation_params(
        ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
        shape_guidance_strength, shape_guidance_rescale, shape_sampling_steps, shape_rescale_t,
        tex_guidance_strength, tex_guidance_rescale, tex_sampling_steps, tex_rescale_t,
    )
    # View labels are returned/stored for organization only. TRELLIS.2's released
    # image-conditioning model consumes image tokens and does not consume camera labels.
    return await _start_generation(images, labels, task_id, seed, profile, resolution, params)


# ---------------------------------------------------------------------------
# API: Extract GLB
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# API: Extract GLB
# ---------------------------------------------------------------------------
@app.post("/api/extract-glb")
async def extract_glb(
    session_id: str = Form(...),
    decimation_target: int = Form(500000),
    texture_size: int = Form(2048),
):
    if session_id not in latent_store:
        raise HTTPException(status_code=404, detail="Session not found. Generate a model first.")

    state = latent_store[session_id]
    session_dir = get_session_dir(session_id)
    meter = VramMeter(torch, MESH_DEVICE)
    profile = resolve_profile(state.get("preset", DEFAULT_PROFILE_REQUEST), GPU_INFO)

    async with gpu_lock:
        try:
            def do_export():
                import torch
                import gc

                # Model weights stay on CPU while the PBR mesh is unwrapped and baked.
                with torch.no_grad():
                    force_offload_all()

                dev = MESH_DEVICE
                with torch.no_grad(), torch.cuda.device(dev):
                    meter.begin("GLB export: remesh and PBR bake")
                    allocated = torch.cuda.memory_allocated(dev) / 1e9
                    print(f"\n[GLB] Before export on {dev}: {allocated:.2f} GB allocated.")

                    verts = state['verts'].to(dev)
                    faces = state['faces'].to(dev)
                    attrs = state['attrs'].to(dev)
                    coords = state['coords'].to(dev)

                    print(f"[GLB] Exporting on {dev}, decimation={decimation_target}, tex_size={texture_size}")

                    try:
                        glb = o_voxel.postprocess.to_glb(
                            vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                            attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target, texture_size=texture_size,
                            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
                        )
                        meter.end()
                    except Exception as e:
                        if is_cuda_oom(e, torch):
                            meter.end()
                            print(f"⚠ OOM detected in to_glb with remesh=True! Retrying with remesh=False (Fallback Mode)...")
                            # Wipe the failed tensors and free memory
                            del verts, faces, attrs, coords
                            gc.collect()
                            torch.cuda.empty_cache()

                            # Reload pure CPU tensors to GPU for the retry
                            verts = state['verts'].to(dev)
                            faces = state['faces'].to(dev)
                            attrs = state['attrs'].to(dev)
                            coords = state['coords'].to(dev)

                            # Retry without the memory-intensive cumesh remeshing pass
                            meter.begin("GLB export fallback: no remesh")
                            glb = o_voxel.postprocess.to_glb(
                                vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                                attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                decimation_target=decimation_target, texture_size=texture_size,
                                remesh=False, use_tqdm=True,
                            )
                            meter.end()
                        else:
                            raise e

                del verts, faces, attrs, coords
                gc.collect()
                torch.cuda.empty_cache()

                ts = make_timestamp()
                glb_path = session_dir / f"sample_{ts}.glb"
                glb.export(str(glb_path))

                del glb
                gc.collect()
                return str(glb_path), meter.peak_gb

            glb_path, peak_gb = await asyncio.get_event_loop().run_in_executor(None, do_export)

            headers = {"X-Trellis-Peak-VRAM-GB": f"{peak_gb:.3f}"} if peak_gb is not None else {}
            return FileResponse(glb_path, media_type="model/gltf-binary", filename=Path(glb_path).name, headers=headers)

        except Exception as e:
            try:
                force_offload_all()
            finally:
                _empty_cuda_cache()
            if is_cuda_oom(e, torch):
                try:
                    meter.end()
                except Exception:
                    pass
                detail = oom_detail(meter.stage, meter.peak_gb, profile, state.get("res"))
                raise HTTPException(status_code=500, detail=detail) from None
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from None


# ---------------------------------------------------------------------------
# API: Extract OBJ (ZIP with .obj + .mtl + textures)
# ---------------------------------------------------------------------------
@app.post("/api/extract-obj")
async def extract_obj(
    session_id: str = Form(...),
    model_name: str = Form("model"),
    decimation_target: int = Form(500000),
    texture_size: int = Form(2048),
):
    model_name = sanitize_model_name(model_name)

    if session_id not in latent_store:
        raise HTTPException(status_code=404, detail="Session not found. Generate a model first.")

    state = latent_store[session_id]
    session_dir = get_session_dir(session_id)
    meter = VramMeter(torch, MESH_DEVICE)
    profile = resolve_profile(state.get("preset", DEFAULT_PROFILE_REQUEST), GPU_INFO)

    async with gpu_lock:
        try:
            def do_export():
                import torch
                import gc

                # Keep inference networks off CUDA while UVs and PBR maps are baked.
                with torch.no_grad():
                    force_offload_all()

                dev = MESH_DEVICE
                with torch.no_grad(), torch.cuda.device(dev):
                    meter.begin("OBJ export: remesh and PBR bake")
                    allocated = torch.cuda.memory_allocated(dev) / 1e9
                    print(f"\n[OBJ] Before export on {dev}: {allocated:.2f} GB allocated.")

                    verts = state['verts'].to(dev)
                    faces = state['faces'].to(dev)
                    attrs = state['attrs'].to(dev)
                    coords = state['coords'].to(dev)

                    print(f"[OBJ] Exporting on {dev}, decimation={decimation_target}, tex_size={texture_size}")
                    try:
                        textured_mesh = o_voxel.postprocess.to_glb(
                            vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                            attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target, texture_size=texture_size,
                            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
                        )
                        meter.end()
                    except Exception as e:
                        if is_cuda_oom(e, torch):
                            meter.end()
                            print(f"⚠ OOM detected in to_glb with remesh=True! Retrying with remesh=False (Fallback Mode)...")
                            del verts, faces, attrs, coords
                            gc.collect()
                            torch.cuda.empty_cache()

                            verts = state['verts'].to(dev)
                            faces = state['faces'].to(dev)
                            attrs = state['attrs'].to(dev)
                            coords = state['coords'].to(dev)

                            meter.begin("OBJ export fallback: no remesh")
                            textured_mesh = o_voxel.postprocess.to_glb(
                                vertices=verts, faces=faces, attr_volume=attrs, coords=coords,
                                attr_layout=state['pbr_layout'], grid_size=state['res'], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                decimation_target=decimation_target, texture_size=texture_size,
                                remesh=False, use_tqdm=True,
                            )
                            meter.end()
                        else:
                            raise e

                del verts, faces, attrs, coords
                gc.collect()
                torch.cuda.empty_cache()

                ts = make_timestamp()
                obj_dir = session_dir / f"obj_{ts}"
                obj_dir.mkdir(exist_ok=True)

                obj_path = obj_dir / f"{model_name}.obj"
                textured_mesh.export(str(obj_path), file_type='obj', include_texture=True)

                # ── Save individual PBR maps alongside OBJ ──
                try:
                    mat = textured_mesh.visual.material
                    # BaseColor (RGB) + Alpha from baseColorTexture (RGBA)
                    bc_tex = getattr(mat, 'baseColorTexture', None)
                    if bc_tex is not None:
                        bc_arr = np.array(bc_tex)
                        Image.fromarray(bc_arr[:, :, :3]).save(str(obj_dir / f"{model_name}_basecolor.png"))
                        if bc_arr.shape[-1] == 4:
                            Image.fromarray(bc_arr[:, :, 3]).save(str(obj_dir / f"{model_name}_alpha.png"))

                    # MetallicRoughness from metallicRoughnessTexture
                    # glTF packing: R=unused, G=roughness, B=metallic
                    mr_tex = getattr(mat, 'metallicRoughnessTexture', None)
                    if mr_tex is not None:
                        mr_arr = np.array(mr_tex)
                        Image.fromarray(mr_arr[:, :, 2]).save(str(obj_dir / f"{model_name}_metallic.png"))
                        Image.fromarray(mr_arr[:, :, 1]).save(str(obj_dir / f"{model_name}_roughness.png"))

                    # Normal map — baked by patched o_voxel via nvdiffrast (GPU)
                    nm_tex = getattr(mat, 'normalTexture', None)
                    if nm_tex is not None:
                        Image.fromarray(np.array(nm_tex)[:, :, :3]).save(str(obj_dir / f"{model_name}_normal.png"))
                        print(f"[OBJ] Normal map extracted from PBR material (GPU-baked)")
                    else:
                        # Fallback: bake normal map from mesh vertex normals on CPU
                        # (used if o_voxel was installed without the normal_map_bake patch)
                        try:
                            mesh_vn = textured_mesh.vertex_normals
                            mesh_uv = textured_mesh.visual.uv
                            mesh_faces = textured_mesh.faces
                            tex_h = bc_arr.shape[0] if bc_tex is not None else texture_size
                            tex_w = bc_arr.shape[1] if bc_tex is not None else texture_size

                            nmap = np.full((tex_h, tex_w, 3), [128, 128, 255], dtype=np.uint8)
                            wmap = np.zeros((tex_h, tex_w), dtype=np.float64)
                            nmap_f = np.zeros((tex_h, tex_w, 3), dtype=np.float64)

                            for fi in range(mesh_faces.shape[0]):
                                vi = mesh_faces[fi]
                                uv_tri = mesh_uv[vi]
                                fn = mesh_vn[vi].mean(axis=0)
                                fn = fn / (np.linalg.norm(fn) + 1e-8)

                                px = (uv_tri[:, 0] * (tex_w - 1)).astype(np.int32)
                                py = ((1 - uv_tri[:, 1]) * (tex_h - 1)).astype(np.int32)
                                pts = np.stack([px, py], axis=-1)

                                x0, y0 = pts.min(axis=0).clip(0)
                                x1, y1 = np.minimum(pts.max(axis=0), [tex_w-1, tex_h-1])
                                if x0 >= x1 or y0 >= y1:
                                    continue

                                m = np.zeros((y1-y0+1, x1-x0+1), dtype=np.uint8)
                                cv2.fillConvexPoly(m, pts - [[x0, y0]], 1)
                                sel = m > 0
                                nmap_f[y0:y1+1, x0:x1+1][sel] += fn
                                wmap[y0:y1+1, x0:x1+1][sel] += 1.0

                            valid = wmap > 0
                            for c in range(3):
                                nmap_f[:,:,c][valid] /= wmap[valid]
                            nrm_len = np.maximum(np.linalg.norm(nmap_f, axis=-1, keepdims=True), 1e-8)
                            nmap_f = nmap_f / nrm_len
                            nmap_f[~valid] = [0, 0, 1]
                            nmap = np.clip((nmap_f * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
                            mask_inv_nm = (~valid).astype(np.uint8)
                            nmap = cv2.inpaint(nmap, mask_inv_nm, 3, cv2.INPAINT_TELEA)

                            Image.fromarray(nmap).save(str(obj_dir / f"{model_name}_normal.png"))
                            print(f"[OBJ] Normal map baked from vertex normals (CPU fallback, {tex_w}x{tex_h})")
                        except Exception as nm_err:
                            print(f"⚠ Could not bake normal map: {nm_err}")
                            import traceback as _tb; _tb.print_exc()

                    print(f"[OBJ] PBR maps saved for '{model_name}': basecolor, metallic, roughness, normal, alpha")
                except Exception as pbr_err:
                    print(f"⚠ Could not save PBR maps: {pbr_err}")

                # ── Write comprehensive .mtl referencing all PBR maps ──
                try:
                    mtl_path = obj_dir / f"{model_name}.mtl"
                    mtl_lines = [
                        f"# PBR Material for {model_name}",
                        f"# Exported by TRELLIS.2 WebUI",
                        f"newmtl {model_name}",
                        "Ka 0.2 0.2 0.2",
                        "Kd 1.0 1.0 1.0",
                        "Ks 0.5 0.5 0.5",
                        "Ns 225.0",
                        "d 1.0",
                        "illum 2",
                        f"map_Kd {model_name}_basecolor.png",
                    ]
                    if (obj_dir / f"{model_name}_metallic.png").exists():
                        mtl_lines.append(f"map_Pm {model_name}_metallic.png")       # PBR metallic
                        mtl_lines.append(f"refl -type sphere {model_name}_metallic.png")  # Fallback
                    if (obj_dir / f"{model_name}_roughness.png").exists():
                        mtl_lines.append(f"map_Pr {model_name}_roughness.png")      # PBR roughness
                        mtl_lines.append(f"map_Ns {model_name}_roughness.png")      # Fallback: roughness → specular exponent map
                    if (obj_dir / f"{model_name}_normal.png").exists():
                        mtl_lines.append(f"map_Bump {model_name}_normal.png")       # Normal/bump map
                        mtl_lines.append(f"norm {model_name}_normal.png")            # Explicit normal map (modern .mtl)
                    if (obj_dir / f"{model_name}_alpha.png").exists():
                        mtl_lines.append(f"map_d {model_name}_alpha.png")            # Dissolve/opacity map
                    mtl_lines.append("")
                    mtl_path.write_text("\n".join(mtl_lines))

                    # Patch OBJ to reference our mtl (trimesh may write its own)
                    obj_text = obj_path.read_text()
                    if "mtllib" in obj_text:
                        obj_text = "\n".join(
                            line if not line.startswith("mtllib") else f"mtllib {model_name}.mtl"
                            for line in obj_text.splitlines()
                        )
                    else:
                        obj_text = f"mtllib {model_name}.mtl\n" + obj_text
                    # Ensure usemtl references our material name
                    if "usemtl" in obj_text:
                        import re as _re
                        obj_text = _re.sub(r'usemtl\s+\S+', f'usemtl {model_name}', obj_text)
                    else:
                        obj_text = obj_text.replace(f"mtllib {model_name}.mtl", f"mtllib {model_name}.mtl\nusemtl {model_name}", 1)
                    obj_path.write_text(obj_text)

                    print(f"[OBJ] Wrote {model_name}.mtl with all PBR map references")
                except Exception as mtl_err:
                    print(f"⚠ Could not write material.mtl: {mtl_err}")

                del textured_mesh
                gc.collect()

                # Remove any stale trimesh-generated .mtl and texture files (we wrote our own)
                for stale_mtl in obj_dir.glob("*.mtl"):
                    if stale_mtl.name != f"{model_name}.mtl":
                        stale_mtl.unlink(missing_ok=True)
                # Remove trimesh's auto-generated texture (typically material_0.png)
                for stale_tex in obj_dir.glob("material_*.png"):
                    stale_tex.unlink(missing_ok=True)

                zip_path = session_dir / f"{model_name}.zip"
                with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                    for f in obj_dir.iterdir():
                        zf.write(str(f), f.name)

                shutil.rmtree(str(obj_dir), ignore_errors=True)
                return str(zip_path), meter.peak_gb

            zip_path, peak_gb = await asyncio.get_event_loop().run_in_executor(None, do_export)

            headers = {"X-Trellis-Peak-VRAM-GB": f"{peak_gb:.3f}"} if peak_gb is not None else {}
            return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name, headers=headers)

        except Exception as e:
            try:
                force_offload_all()
            finally:
                _empty_cuda_cache()
            if is_cuda_oom(e, torch):
                try:
                    meter.end()
                except Exception:
                    pass
                detail = oom_detail(meter.stage, meter.peak_gb, profile, state.get("res"))
                raise HTTPException(status_code=500, detail=detail) from None
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e)) from None


# ---------------------------------------------------------------------------
# API: Convert GLB to OBJ (no GPU required)
# ---------------------------------------------------------------------------
@app.post("/api/convert-to-obj")
async def convert_to_obj(glb_file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = get_session_dir(session_id)

    try:
        glb_data = await glb_file.read()
        glb_tmp = session_dir / "input.glb"
        glb_tmp.write_bytes(glb_data)

        def do_convert():
            mesh = trimesh.load(str(glb_tmp))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.to_mesh()

            ts = make_timestamp()
            obj_dir = session_dir / f"obj_{ts}"
            obj_dir.mkdir(exist_ok=True)

            obj_path = obj_dir / "model.obj"
            mesh.export(str(obj_path), file_type='obj', include_texture=True)

            zip_path = session_dir / f"model_{ts}.zip"
            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in obj_dir.iterdir():
                    zf.write(str(f), f.name)

            shutil.rmtree(str(obj_dir), ignore_errors=True)
            return str(zip_path)

        zip_path = await asyncio.get_event_loop().run_in_executor(None, do_convert)
        return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Re-Texturing
# ---------------------------------------------------------------------------
@app.post("/api/texturing")
async def texturing(
    mesh_file: UploadFile = File(...),
    image: UploadFile = File(...),
    seed: int = Form(11456),
    randomize_seed: bool = Form(False),
    resolution: str = Form("1024"),
    execution_preset: str = Form(""),
    texture_size: int = Form(2048),
    tex_guidance_strength: float = Form(1.0),
    tex_guidance_rescale: float = Form(0.0),
    tex_sampling_steps: int = Form(12),
    tex_rescale_t: float = Form(3.0),
):
    if randomize_seed:
        seed = int(np.random.randint(0, MAX_SEED))
    profile = resolve_profile(execution_preset or DEFAULT_PROFILE_REQUEST, GPU_INFO)
    try:
        offload = (
            True
            if not _LOAD_HIGH_RESOLUTION_MODELS
            else resolve_cpu_offload(CPU_OFFLOAD_REQUEST, profile, GPU_INFO.get("total_vram_gb"))
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    # The upstream re-texturing pipeline supports 512 and 1024 only. A selected
    # 1536 generation preset therefore uses its supported 1024 texture path.
    requested_texture_resolution = str(resolution)
    texture_resolution = (
        512 if not _LOAD_HIGH_RESOLUTION_MODELS or requested_texture_resolution == "512" else 1024
    )
    texture_resolution_warning = (
        "Low-memory mode uses 512 texture resolution; 1024 texture weights are not loaded."
        if not _LOAD_HIGH_RESOLUTION_MODELS and requested_texture_resolution != "512"
        else None
    )

    session_id = str(uuid.uuid4())
    session_dir = get_session_dir(session_id)

    mesh_data = await mesh_file.read()
    suffix = Path(mesh_file.filename or "mesh.glb").suffix
    mesh_tmp = session_dir / f"input_mesh{suffix}"
    mesh_tmp.write_bytes(mesh_data)

    mesh_obj = trimesh.load(str(mesh_tmp))
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = mesh_obj.to_mesh()

    pil_image = load_pil_image(image)

    async with gpu_lock:
        try:
            meter = VramMeter(torch, GENERATION_DEVICE)
            def do_texture():
                global tex_pipeline
                output = None
                prepared_image = None
                stage_open = False
                try:
                    force_offload_all()
                    if tex_pipeline is None:
                        print("Loading re-texturing components (sharing generation texture models)...", flush=True)
                        tex_pipeline = load_texturing_pipeline_shared()
                    meter.begin("Re-texturing model preparation")
                    stage_open = True
                    configure_pipeline_for_job(tex_pipeline, offload)
                    meter.end()
                    stage_open = False
                    with torch.no_grad():
                        meter.begin("Re-texturing: conditioning, texture sampling, decode and bake")
                        stage_open = True
                        prepared_image = tex_pipeline.preprocess_image(pil_image)
                        output = tex_pipeline.run(
                            mesh_obj, prepared_image, seed=seed, preprocess_image=False,
                            tex_slat_sampler_params={"steps": tex_sampling_steps, "guidance_strength": tex_guidance_strength, "guidance_rescale": tex_guidance_rescale, "rescale_t": tex_rescale_t},
                            resolution=texture_resolution, texture_size=texture_size,
                        )
                        meter.end()
                        stage_open = False
                    ts = make_timestamp()
                    glb_path = session_dir / f"textured_{ts}.glb"
                    output.export(str(glb_path))

                    return str(glb_path), meter.peak_gb
                except Exception as exc:
                    if is_cuda_oom(exc, torch):
                        if stage_open and meter.available:
                            try:
                                meter.end()
                            except Exception:
                                pass
                        detail = oom_detail(meter.stage, meter.peak_gb, profile, texture_resolution)
                        try:
                            exc.__traceback__ = None
                            exc.__context__ = None
                        except Exception:
                            pass
                        raise GenerationOOMError(detail) from None
                    raise
                finally:
                    output = prepared_image = None
                    try:
                        force_offload_all()
                    except Exception as cleanup_error:
                        print(f"[VRAM] Re-texturing cleanup warning: {cleanup_error}", flush=True)
                    if not _LOAD_HIGH_RESOLUTION_MODELS:
                        tex_pipeline = None
                        release_pipeline_models((
                            "tex_slat_decoder",
                            "tex_slat_flow_model_512",
                            "tex_slat_flow_model_1024",
                        ))
                    _empty_cuda_cache()

            glb_path, peak_gb = await asyncio.get_event_loop().run_in_executor(None, do_texture)
            headers = {"X-Trellis-Peak-VRAM-GB": f"{peak_gb:.3f}"} if peak_gb is not None else {}
            if texture_resolution_warning:
                headers["X-Trellis-Resolution-Warning"] = texture_resolution_warning
            return FileResponse(glb_path, media_type="model/gltf-binary", filename=Path(glb_path).name, headers=headers)

        except Exception as e:
            if isinstance(e, GenerationOOMError):
                raise HTTPException(status_code=500, detail=str(e)) from None
            try:
                force_offload_all()
            finally:
                _empty_cuda_cache()
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: Example images list
# ---------------------------------------------------------------------------
@app.get("/api/examples")
async def list_examples():
    examples_dir = ASSETS_DIR / "example_image"
    if not examples_dir.exists():
        return {"examples": []}
    files = sorted([f.name for f in examples_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])
    return {"examples": files}


@app.get("/api/examples/{filename}")
async def get_example(filename: str):
    filepath = ASSETS_DIR / "example_image" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(filepath))


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
if WEBUI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEBUI_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return {"message": "Frontend not built. Run 'npm run build' in webui/ or use 'npm run dev' for development."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.environ.get("TRELLIS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
