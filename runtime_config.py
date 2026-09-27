"""Runtime profiles and CUDA memory telemetry for the WebUI.

The upstream checkpoint selects the actual per-model weight formats.  This
module does not rewrite model precision; it only chooses a safe resolution and
controls when upstream components are resident on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ExecutionProfile:
    key: str
    label: str
    resolution: str
    sampling_steps: int
    decimation_target: int
    texture_size: int
    warning: str = ""


PROFILES: dict[str, ExecutionProfile] = {
    "16gb-fast": ExecutionProfile(
        "16gb-fast", "16GB Fast", "512", 12, 500_000, 2048,
    ),
    "16gb-quality": ExecutionProfile(
        "16gb-quality", "16GB Quality", "512", 20, 800_000, 2048,
        "512 geometry with more sampling steps; 16 GB fit is not yet benchmark-verified.",
    ),
    "24gb-high": ExecutionProfile(
        "24gb-high", "24GB+ High", "1024", 30, 800_000, 4096,
        "Uses the 1024 cascade. Requires more than 16 GB in some configurations.",
    ),
    "maximum": ExecutionProfile(
        "maximum", "Maximum", "1536", 50, 1_000_000, 4096,
        "Experimental maximum-quality path; more than 16 GB VRAM may be required.",
    ),
}


def recommend_profile(total_vram_gb: float | None) -> str:
    """Choose a conservative UI default from detected device capacity."""
    if total_vram_gb is None:
        return "16gb-fast"
    if total_vram_gb < 24:
        return "16gb-fast"
    if total_vram_gb <= 32:
        return "24gb-high"
    return "maximum"


def get_gpu_info(torch_module: Any) -> dict[str, Any]:
    """Return point-in-time CUDA details without making CUDA a hard requirement."""
    info: dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "total_vram_gb": None,
        "free_vram_gb": None,
        "bf16_supported": False,
    }
    try:
        if not torch_module.cuda.is_available():
            return info
        device = torch_module.device("cuda:0")
        properties = torch_module.cuda.get_device_properties(device)
        total_bytes = int(properties.total_memory)
        info.update({
            "cuda_available": True,
            "gpu_name": properties.name,
            "total_vram_gb": total_bytes / (1024 ** 3),
            "bf16_supported": bool(torch_module.cuda.is_bf16_supported()),
        })
        try:
            free_bytes, _ = torch_module.cuda.mem_get_info(device)
            info["free_vram_gb"] = free_bytes / (1024 ** 3)
        except Exception:
            pass
    except Exception as exc:
        info["detection_error"] = str(exc)
    return info


def resolve_profile(requested: str | None, gpu_info: dict[str, Any]) -> ExecutionProfile:
    key = (requested or "auto").strip().lower()
    if key in ("", "auto"):
        key = recommend_profile(gpu_info.get("total_vram_gb"))
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown execution preset '{requested}'. Choose one of: {', '.join(PROFILES)} or auto.") from exc


def resolve_cpu_offload(
    requested: str | None,
    profile: ExecutionProfile,
    total_vram_gb: float | None = None,
) -> bool:
    """Low-memory profiles always offload in auto mode; high profiles use residency."""
    value = (requested or "auto").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    if value != "auto":
        raise ValueError("TRELLIS_CPU_OFFLOAD must be auto, 0, or 1.")
    if total_vram_gb is None or total_vram_gb < 24:
        return True
    return profile.key in ("16gb-fast", "16gb-quality")


class VramMeter:
    """Stage-boundary CUDA allocator snapshots for one generation request."""

    def __init__(self, torch_module: Any, device: str = "cuda:0", on_snapshot: Callable[[dict[str, Any]], None] | None = None):
        self.torch = torch_module
        self.device = device
        self.on_snapshot = on_snapshot
        self.stage = "starting"
        self.records: list[dict[str, Any]] = []
        self.peak_allocated_bytes = 0
        self.available = bool(torch_module.cuda.is_available())

    def begin(self, stage: str) -> None:
        self.stage = stage
        if not self.available:
            return
        try:
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            # Instrumentation must not make otherwise-valid inference fail.
            pass

    def end(self) -> dict[str, Any]:
        if not self.available:
            return {"stage": self.stage, "available": False}
        try:
            self.torch.cuda.synchronize(self.device)
            allocated = int(self.torch.cuda.memory_allocated(self.device))
            reserved = int(self.torch.cuda.memory_reserved(self.device))
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
        except Exception as exc:
            record = {"stage": self.stage, "available": False, "error": str(exc)}
            self.records.append(record)
            return record

        self.peak_allocated_bytes = max(self.peak_allocated_bytes, peak)
        record = {
            "stage": self.stage,
            "allocated_gb": allocated / (1024 ** 3),
            "reserved_gb": reserved / (1024 ** 3),
            "peak_allocated_gb": peak / (1024 ** 3),
        }
        self.records.append(record)
        print(
            f"[VRAM] {self.stage}: allocated={record['allocated_gb']:.2f} GB "
            f"reserved={record['reserved_gb']:.2f} GB "
            f"peak={record['peak_allocated_gb']:.2f} GB",
            flush=True,
        )
        if self.on_snapshot is not None:
            self.on_snapshot(record)
        return record

    @property
    def peak_gb(self) -> float | None:
        if not self.available:
            return None
        return self.peak_allocated_bytes / (1024 ** 3)


def is_cuda_oom(exc: BaseException, torch_module: Any) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", ())
    return (isinstance(oom_type, type) and isinstance(exc, oom_type)) or (
        "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()
    )


def oom_detail(
    stage: str,
    peak_gb: float | None,
    profile: ExecutionProfile,
    resolution: str | int | None = None,
) -> str:
    peak = f" Peak allocated VRAM: {peak_gb:.1f} GB." if peak_gb is not None else ""
    normalized_stage = stage.lower()
    if "export" in normalized_stage:
        suggestion = "Reduce export texture size and decimation target; the no-remesh fallback has already been attempted."
    elif resolution is not None and str(resolution) != "512":
        suggestion = 'Retry with geometry/texture resolution 512 using "16GB Fast".'
    elif profile.key == "maximum":
        suggestion = 'Try "16GB Fast" at 512 resolution or reduce the sampling steps.'
    elif profile.key == "24gb-high":
        suggestion = 'Try "16GB Fast" at 512 resolution.'
    elif profile.key == "16gb-quality":
        suggestion = 'Try "16GB Fast" or reduce the output texture size.'
    else:
        suggestion = "Close other GPU workloads and retry, or reduce the output texture size."
    return f"CUDA out of memory during {stage}.{peak} {suggestion}"
