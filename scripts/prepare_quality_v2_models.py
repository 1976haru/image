"""Explicitly download Quality Engine V2 runtime + weights (never run during generation).

    python scripts/prepare_quality_v2_models.py [--models-dir D:\\models] [--only sdcpp zimage flux2 qwen3 vae]
    python scripts/prepare_quality_v2_models.py --verify

Everything goes under <models-dir>/quality_v2 (outside git, ~16 GB). Each file is checked against the
SHA-256 recorded on 2026-10-04; a mismatch is deleted and reported, never used. Downloads resume.
Check free disk space first: the script refuses to start with less than 20 GiB free.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.quality_engines import SDCPP_RELEASE  # noqa: E402

GH = f"https://github.com/leejet/stable-diffusion.cpp/releases/download/{SDCPP_RELEASE}"
HF = "https://huggingface.co"
# group, relative path, url, sha256, license note
FILES = [
    ("sdcpp", "sdcpp/sd-cuda12.zip", f"{GH}/sd-master-3f8527a-bin-win-cuda12-x64.zip",
     "217d6dead9abd3f827fc338268555cc179234e7e6330ef21ecb1c985e19d2dc7", "MIT (stable-diffusion.cpp)"),
    ("sdcpp", "sdcpp/cudart.zip", f"{GH}/cudart-sd-bin-win-cu12-x64.zip",
     "fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d", "NVIDIA CUDA runtime redistributable"),
    ("zimage", "zimage_turbo/z_image_turbo-Q6_K.gguf", f"{HF}/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q6_K.gguf",
     "319f627beac8059b7546f36a7b4d5097b7f4ee6a1fc37585d0f75ca1d12d01af", "Apache-2.0 (Tongyi-MAI/Z-Image-Turbo)"),
    ("flux2", "flux2_klein_4b/flux-2-klein-4b-Q8_0.gguf",
     f"{HF}/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q8_0.gguf",
     "0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28", "Apache-2.0 (black-forest-labs/FLUX.2-klein-4B)"),
    ("qwen3", "qwen3_4b/Qwen3-4B-Q8_0.gguf", f"{HF}/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
     "eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d", "Apache-2.0 (Qwen/Qwen3-4B)"),
    ("vae", "vae/flux1_ae.safetensors", f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
     "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38", "Apache-2.0 (FLUX.1 VAE, shipped with Z-Image-Turbo)"),
    ("vae", "vae/flux2_ae.safetensors", f"{HF}/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
     "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
     "Apache-2.0 (FLUX.2 VAE; the same autoencoder ships in black-forest-labs/FLUX.2-klein-4B)"),
]
MIN_FREE_GIB = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        mode = "ab" if start and response.status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, 16 * 1024 * 1024)
    os.replace(partial, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path(os.environ.get("COVERMORPH_MODELS_DIR") or REPOSITORY_ROOT / "models"))
    parser.add_argument("--only", nargs="*", choices=sorted({group for group, *_ in FILES}))
    parser.add_argument("--verify", action="store_true", help="only check hashes of files already present")
    args = parser.parse_args()
    root = args.models_dir / "quality_v2"
    root.mkdir(parents=True, exist_ok=True)
    if not args.verify and shutil.disk_usage(root).free < MIN_FREE_GIB * 1024 ** 3:
        print(f"Refusing to download: less than {MIN_FREE_GIB} GiB free on {root}.")
        return 2
    failures = 0
    for group, relative, url, expected, license_note in FILES:
        if args.only and group not in args.only:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if args.verify:
                print(f"MISSING {relative}")
                failures += 1
                continue
            print(f"Downloading {relative}  [{license_note}]", flush=True)
            download(url, target)
        actual = sha256(target)
        if actual != expected:
            print(f"HASH MISMATCH {relative}: {actual} (expected {expected}); file removed")
            if not args.verify:
                target.unlink()
            failures += 1
            continue
        print(f"OK {relative}")
        if target.suffix == ".zip" and not args.verify:
            with zipfile.ZipFile(target) as archive:
                archive.extractall(target.parent)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
