#!/usr/bin/env python3
"""Run one WebUI generation and GLB export, printing measured VRAM/time/size.

The WebUI server must already be running with its TRELLIS.2 model loaded. Pass
one image for ``--views 1`` and exactly 2, 4, or 6 images for multi-view runs.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from runtime_config import PROFILES


def encode_multipart(fields: list[tuple[str, str]], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = f"----trellis2-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields:
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    for name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            path.read_bytes(),
            b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def post_multipart(url: str, fields: list[tuple[str, str]], files: list[tuple[str, Path]], timeout: int):
    body, boundary = encode_multipart(fields, files)
    request = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.headers, response.read()


def pick_images(args) -> list[Path]:
    images = [Path(path).expanduser().resolve() for path in args.image]
    if not images:
        examples = Path(__file__).parent / "assets" / "example_image"
        if examples.exists():
            images = sorted(
                path.resolve() for path in examples.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )[:args.views]
    if len(images) != args.views:
        raise ValueError(f"--views {args.views} requires exactly {args.views} --image path(s); found {len(images)}.")
    for path in images:
        if not path.is_file():
            raise ValueError(f"Image not found: {path}")
    return images


def response_error(error: HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        return str(payload.get("detail", payload))
    except Exception:
        return f"HTTP {error.code}: {error.reason}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, choices=(1, 2, 4, 6), required=True)
    parser.add_argument("--preset", choices=tuple(PROFILES), default="16gb-fast")
    parser.add_argument("--resolution", choices=("512", "1024", "1536"), help="Override the preset geometry resolution")
    parser.add_argument("--image", action="append", default=[], help="Input image path; repeat once per view")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    try:
        images = pick_images(args)
        profile = PROFILES[args.preset]
        resolution = args.resolution or profile.resolution
    except ValueError as exc:
        parser.error(str(exc))

    task_id = str(uuid.uuid4())
    fields = [
        ("seed", str(args.seed)),
        ("randomize_seed", "false"),
        ("resolution", resolution),
        ("execution_preset", profile.key),
        ("task_id", task_id),
        ("ss_guidance_strength", "7.5"),
        ("ss_guidance_rescale", "0.7"),
        ("ss_sampling_steps", str(profile.sampling_steps)),
        ("ss_rescale_t", "5.0"),
        ("shape_guidance_strength", "7.5"),
        ("shape_guidance_rescale", "0.5"),
        ("shape_sampling_steps", str(profile.sampling_steps)),
        ("shape_rescale_t", "3.0"),
        ("tex_guidance_strength", "1.0"),
        ("tex_guidance_rescale", "0.0"),
        ("tex_sampling_steps", str(profile.sampling_steps)),
        ("tex_rescale_t", "3.0"),
    ]
    if args.views == 1:
        endpoint = "/api/generate"
        files = [("image", images[0])]
    else:
        endpoint = "/api/generate-multiview"
        files = []
        for index, path in enumerate(images):
            files.append((f"image_{index}", path))
            fields.append((f"view_label_{index}", "Custom"))

    result = {
        "preset": profile.key,
        "resolution_requested": resolution,
        "resolution": resolution,
        "views": args.views,
        "input_images": [str(path) for path in images],
        "generation_success": False,
        "generation_seconds": None,
        "peak_vram_gb": None,
        "glb_export_peak_vram_gb": None,
        "glb_size_bytes": None,
    }
    base_url = args.base_url.rstrip("/")
    generation_start = None
    export_start = None
    try:
        generation_start = time.perf_counter()
        status, _, payload = post_multipart(base_url + endpoint, fields, files, args.timeout)
        generation_seconds = time.perf_counter() - generation_start
        generation = json.loads(payload.decode("utf-8"))
        result.update({
            "generation_success": 200 <= status < 300,
            "generation_seconds": round(generation_seconds, 3),
            "resolution": generation.get("resolution", resolution),
            "peak_vram_gb": generation.get("peak_vram_gb"),
            "vram_stages": generation.get("vram_stages", []),
            "warning": generation.get("warning"),
        })
        session_id = generation["session_id"]
        export_fields = [
            ("session_id", session_id),
            ("decimation_target", str(profile.decimation_target)),
            ("texture_size", str(profile.texture_size)),
        ]
        export_start = time.perf_counter()
        _, export_headers, glb_bytes = post_multipart(
            base_url + "/api/extract-glb", export_fields, [], args.timeout
        )
        result["glb_export_seconds"] = round(time.perf_counter() - export_start, 3)
        export_peak = export_headers.get("X-Trellis-Peak-VRAM-GB")
        result["glb_export_peak_vram_gb"] = float(export_peak) if export_peak is not None else None
        result["glb_size_bytes"] = len(glb_bytes)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f"{profile.key}_{args.views}views_{resolution}.glb"
        output.write_bytes(glb_bytes)
        result["glb_path"] = str(output.resolve())
        result["success"] = True
    except HTTPError as exc:
        detail = response_error(exc)
        result["error"] = detail
        result["http_status"] = exc.code
        if generation_start is not None and not result["generation_success"]:
            result["generation_seconds"] = round(time.perf_counter() - generation_start, 3)
            peak_match = re.search(r"Peak allocated VRAM:\s*([0-9.]+)\s*GB", detail, re.IGNORECASE)
            stage_match = re.search(r"CUDA out of memory during (.+?)(?:\.\s*Peak allocated VRAM:|\.$)", detail, re.IGNORECASE)
            if peak_match:
                result["peak_vram_gb"] = float(peak_match.group(1))
            if stage_match:
                result["failed_stage"] = stage_match.group(1)
        if export_start is not None and result["generation_success"]:
            result["glb_export_seconds"] = round(time.perf_counter() - export_start, 3)
    except (URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
