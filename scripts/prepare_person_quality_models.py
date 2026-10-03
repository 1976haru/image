"""Explicitly download the external person-quality models (never run during generation).

    python scripts/prepare_person_quality_models.py [--models-dir D:\\models] [--only photoreal|yunet]

Downloads only the diffusers fp16 variant of the photoreal SDXL model and the YuNet face
detector into the external models directory, pinned to the revisions recorded in
covermorph/person_quality.py. Weights are never committed to git.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from covermorph.person_quality import (  # noqa: E402
    MODEL_PROFILES,
    YUNET_FILENAME,
    YUNET_REPOSITORY,
    YUNET_REVISION,
    inspect_photoreal_model,
    inspect_yunet,
)


def prepare_photoreal(models_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    profile = MODEL_PROFILES["photoreal_sdxl"]
    destination = models_dir / profile["folder"]
    print(f"Downloading {profile['repository']}@{profile['revision']} (fp16 variant) -> {destination}", flush=True)
    snapshot_download(
        repo_id=profile["repository"], revision=profile["revision"], local_dir=str(destination),
        allow_patterns=["model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
                        "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
                        "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
                        "README.md"])
    inspection = inspect_photoreal_model(destination)
    print(f"photoreal ready={inspection['ready']} reason={inspection['failure_reason']}", flush=True)


def prepare_yunet(models_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    destination = models_dir / "face_detection"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (YUNET_FILENAME, "LICENSE"):
        hf_hub_download(YUNET_REPOSITORY, name, revision=YUNET_REVISION, local_dir=str(destination))
    print(f"YuNet ready={inspect_yunet(models_dir)['ready']} -> {destination}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path,
                        default=Path(os.environ.get("COVERMORPH_MODELS_DIR") or REPOSITORY_ROOT / "models"))
    parser.add_argument("--only", choices=("photoreal", "yunet"))
    args = parser.parse_args()
    if args.only in (None, "yunet"):
        prepare_yunet(args.models_dir)
    if args.only in (None, "photoreal"):
        prepare_photoreal(args.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
