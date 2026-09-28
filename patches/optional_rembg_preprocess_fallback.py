"""Patch TRELLIS.2 preprocessing for installations without gated RMBG weights.

setup_local.sh deliberately allows the optional background-removal model to
fail to load (for example, when the Hugging Face gate is unavailable). The
upstream preprocessors still unconditionally call ``rembg_model.to(...)`` for
opaque inputs, turning that optional dependency into a request-time crash.

When RMBG is absent, keep the input fully opaque and continue through the
upstream crop/normalization path. This does not remove a photo background;
the WebUI reports that limitation to users. Existing transparent-mask behavior
is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


OLD_BLOCK = '''            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()'''

NEW_BLOCK = '''            if self.rembg_model is None:
                # [WebUI fallback] Preserve opaque input when gated RMBG weights are unavailable.
                # This accepts the image as-is; it does not remove the background.
                output = input.convert('RGBA')
            else:
                if self.low_vram:
                    self.rembg_model.to(self.device)
                try:
                    output = self.rembg_model(input)
                finally:
                    if self.low_vram:
                        self.rembg_model.cpu()'''


def patch(trellis_dir: str | Path) -> None:
    pipelines_dir = Path(trellis_dir) / "trellis2" / "pipelines"
    for filename in ("trellis2_image_to_3d.py", "trellis2_texturing.py"):
        target = pipelines_dir / filename
        if not target.exists():
            print(f"  [SKIP] {target} not found")
            continue

        source = target.read_text(encoding="utf-8")
        if "[WebUI fallback] Preserve opaque input" in source:
            print(f"  [SKIP] opaque-input fallback already applied to {filename}")
            continue
        if OLD_BLOCK not in source:
            raise RuntimeError(
                f"Could not find expected RMBG preprocessing block in {target}; "
                "inspect upstream changes before applying this compatibility patch."
            )

        target.write_text(source.replace(OLD_BLOCK, NEW_BLOCK, 1), encoding="utf-8")
        print(f"  Patched {filename}: opaque input works without RMBG")


if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRELLIS.2")
