from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


class AIBackends:
    def __init__(self, app_root: Path):
        self.app_root = app_root
        self._lama = None
        self._sdxl = None

    def lama_available(self) -> bool:
        try:
            import simple_lama_inpainting  # noqa: F401

            return True
        except Exception:
            return False

    def _runtime_roots(self) -> list[Path]:
        roots = [self.app_root, Path(sys.executable).resolve().parent, Path.cwd()]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))

        unique_roots: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root
            if resolved not in seen:
                unique_roots.append(root)
                seen.add(resolved)
        return unique_roots

    def _tools_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        for root in self._runtime_roots():
            dirs.append(root / "tools")
            dirs.append(root / "tools" / "realesrgan-ncnn-vulkan")
        return dirs

    def realesrgan_exe(self) -> Path | None:
        names = ["realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"]
        for directory in self._tools_dirs():
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        found = shutil.which("realesrgan-ncnn-vulkan")
        return Path(found) if found else None

    def realesrgan_model_available(self, exe: Path, model: str) -> bool:
        candidates = [exe.parent, exe.parent / "models"]
        for tools_dir in self._tools_dirs():
            candidates.extend([tools_dir, tools_dir / "models"])

        for directory in candidates:
            if (directory / f"{model}.param").is_file() and (directory / f"{model}.bin").is_file():
                return True
        return False

    def realesrgan_available(self) -> bool:
        return self.realesrgan_exe() is not None

    def cuda_available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def rembg_available(self) -> bool:
        try:
            import rembg  # noqa: F401

            return True
        except Exception:
            return False

    def sdxl_available(self) -> bool:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    def status(self) -> dict[str, bool]:
        return {
            "lama": self.lama_available(),
            "realesrgan": self.realesrgan_available(),
            "cuda": self.cuda_available(),
            "rembg": self.rembg_available(),
            "sdxl": self.sdxl_available(),
        }

    def person_mask(self, img: Image.Image) -> tuple[Image.Image, str]:
        if not self.rembg_available():
            raise RuntimeError("rembg unavailable")
        from rembg import remove

        rgba = remove(img.convert("RGBA"), only_mask=False)
        return rgba.getchannel("A").convert("L"), "rembg/U2Net"

    def _load_sdxl(self, model_id: str):
        if not self.sdxl_available():
            raise RuntimeError("SDXL dependencies unavailable")
        import torch
        from diffusers import AutoPipelineForInpainting

        if self._sdxl is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            kwargs = {"torch_dtype": dtype, "use_safetensors": True}
            if torch.cuda.is_available():
                kwargs["variant"] = "fp16"
            self._sdxl = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)
            self._sdxl.enable_attention_slicing()
            self._sdxl.to("cuda" if torch.cuda.is_available() else "cpu")
        return self._sdxl

    def outpaint(
        self,
        canvas: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int = 28,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ) -> tuple[Image.Image, str]:
        pipe = self._load_sdxl(model_id)
        if canvas.width >= canvas.height:
            work_size = (
                1024,
                max(512, round(1024 * canvas.height / canvas.width / 64) * 64),
            )
        else:
            work_size = (
                max(512, round(1024 * canvas.width / canvas.height / 64) * 64),
                1024,
            )
        work = canvas.convert("RGB").resize(work_size, Image.Resampling.LANCZOS)
        work_mask = mask.convert("L").resize(work_size, Image.Resampling.NEAREST)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=work,
            mask_image=work_mask,
            num_inference_steps=max(10, min(60, int(steps))),
            guidance_scale=7.0,
        ).images[0]
        return result.resize(canvas.size, Image.Resampling.LANCZOS).convert("RGB"), "SDXL Outpainting"

    def inpaint(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, str]:
        if not self.lama_available():
            raise RuntimeError("LaMa unavailable")
        from simple_lama_inpainting import SimpleLama

        if self._lama is None:
            self._lama = SimpleLama()
        result = self._lama(img.convert("RGB"), mask.convert("L"))
        return result.convert("RGB"), "LaMa"

    def upscale(
        self,
        img: Image.Image,
        scale: int = 2,
        model: str = "realesrgan-x4plus",
    ) -> tuple[Image.Image, str]:
        exe = self.realesrgan_exe()
        if exe is None:
            raise RuntimeError("Real-ESRGAN NCNN executable unavailable")
        if not self.realesrgan_model_available(exe, model):
            raise FileNotFoundError(
                f"Real-ESRGAN model files not found for {model}. "
                "Expected .param and .bin files next to the executable or in a models folder."
            )

        with tempfile.TemporaryDirectory(prefix="covermorph_esrgan_") as temp_name:
            temp_dir = Path(temp_name)
            src = temp_dir / "input.png"
            dst = temp_dir / "output.png"
            img.save(src, "PNG")

            cmd = [
                str(exe),
                "-i",
                str(src),
                "-o",
                str(dst),
                "-n",
                model,
                "-s",
                str(scale),
                "-f",
                "png",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(exe.parent),
                env=os.environ.copy(),
                check=False,
            )
            if proc.returncode != 0 or not dst.exists():
                raise RuntimeError((proc.stderr or proc.stdout or "Real-ESRGAN failed")[-1500:])
            return Image.open(dst).convert("RGB"), f"Real-ESRGAN {model}"
