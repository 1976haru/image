from __future__ import annotations
from pathlib import Path
from typing import Tuple
import shutil
import subprocess
import tempfile
from PIL import Image

class AIBackends:
    def __init__(self, app_root: Path):
        self.app_root = app_root
        self.tools_dir = app_root / "tools"
        self._lama = None
        self._sdxl = None

    def lama_available(self) -> bool:
        try:
            import simple_lama_inpainting
            return True
        except Exception:
            return False

    def realesrgan_exe(self):
        candidates = [
            self.tools_dir / "realesrgan-ncnn-vulkan.exe",
            self.tools_dir / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe",
        ]
        for p in candidates:
            if p.exists():
                return p
        found = shutil.which("realesrgan-ncnn-vulkan")
        return Path(found) if found else None

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
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            return True
        except Exception:
            return False

    def status(self) -> dict:
        return {
            "lama": self.lama_available(),
            "realesrgan": self.realesrgan_available(),
            "cuda": self.cuda_available(),
            "rembg": self.rembg_available(),
            "sdxl": self.sdxl_available(),
        }

    def person_mask(self, img: Image.Image) -> Tuple[Image.Image, str]:
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

    def outpaint(self, canvas: Image.Image, mask: Image.Image, prompt: str,
                 negative_prompt: str, steps: int = 28,
                 model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
        pipe = self._load_sdxl(model_id)
        # Keep the target aspect ratio and use dimensions divisible by 64.
        if canvas.width >= canvas.height:
            work_size = (1024, max(512, (round(1024 * canvas.height / canvas.width / 64) * 64)))
        else:
            work_size = (max(512, (round(1024 * canvas.width / canvas.height / 64) * 64)), 1024)
        work = canvas.convert("RGB").resize(work_size, Image.Resampling.LANCZOS)
        work_mask = mask.convert("L").resize(work_size, Image.Resampling.NEAREST)
        result = pipe(prompt=prompt, negative_prompt=negative_prompt,
                      image=work, mask_image=work_mask,
                      num_inference_steps=max(10, min(60, int(steps))),
                      guidance_scale=7.0).images[0]
        return result.resize(canvas.size, Image.Resampling.LANCZOS).convert("RGB"), "SDXL Outpainting"

    def inpaint(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, str]:
        if not self.lama_available():
            raise RuntimeError("LaMa unavailable")
        from simple_lama_inpainting import SimpleLama
        if self._lama is None:
            self._lama = SimpleLama()
        result = self._lama(img.convert("RGB"), mask.convert("L"))
        return result.convert("RGB"), "LaMa"

    def upscale(self, img: Image.Image, scale: int = 2, model: str = "realesrgan-x4plus") -> Tuple[Image.Image, str]:
        exe = self.realesrgan_exe()
        if exe is None:
            raise RuntimeError("Real-ESRGAN NCNN executable unavailable")

        with tempfile.TemporaryDirectory(prefix="covermorph_esrgan_") as td:
            td = Path(td)
            src = td / "input.png"
            dst = td / "output.png"
            img.save(src, "PNG")

            cmd = [
                str(exe),
                "-i", str(src),
                "-o", str(dst),
                "-n", model,
                "-s", str(scale),
                "-f", "png",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0 or not dst.exists():
                raise RuntimeError((proc.stderr or proc.stdout or "Real-ESRGAN failed")[-1500:])
            return Image.open(dst).convert("RGB"), f"Real-ESRGAN {model}"
