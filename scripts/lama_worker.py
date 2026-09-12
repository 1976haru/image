"""LaMa worker, deliberately independent of the GUI's package environment."""
import sys
from pathlib import Path

from PIL import Image
from simple_lama_inpainting import SimpleLama


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    with Image.open(directory / "input.png") as source, Image.open(directory / "mask.png") as mask:
        result = SimpleLama()(source.convert("RGB"), mask.convert("L"))
        result.save(directory / "output.png", "PNG")
