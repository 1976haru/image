"""Short-lived OCR process for the product text/logo check.

EasyOCR loads a PyTorch model; in the app process that kept ~4.9 GB commit and a 150 MiB CUDA context after a
product job (measured 2026-10-04). Running it here means the memory is returned when this process exits.

    CoverMorphStudio.exe --ocr-worker <image> [<image> ...]      (packaged)
    python -m covermorph.ocr_worker <image> [<image> ...]         (source)
Prints one JSON list of strings (one per image) on stdout.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(paths: list[str]) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never touch the GPU the engines need
    import easyocr
    import numpy as np
    from PIL import Image

    reader = easyocr.Reader(["en"], gpu=False, download_enabled=False, verbose=False)
    texts = []
    for path in paths:
        try:
            with Image.open(path) as opened:
                results = reader.readtext(np.array(opened.convert("RGB")))
            texts.append(" ".join(t for _, t, conf in results if conf >= 0.5 and len(t.strip()) >= 2))
        except Exception:
            texts.append("")
    sys.stdout.write(json.dumps(texts, ensure_ascii=False))
    return 0


def weights_installed() -> bool:
    model_dir = Path.home() / ".EasyOCR" / "model"
    return (model_dir / "craft_mlt_25k.pth").exists() and (model_dir / "english_g2.pth").exists()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
