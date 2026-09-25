from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from covermorph.presets import PRESETS
from covermorph.processor import (
    build_full_frame_outpaint_canvas,
    build_outpaint_canvas,
    detect_text_boxes_multilang,
    inpaint_text_opencv,
    make_fit_original,
    make_shorts,
    make_smart_crop,
    make_square,
    make_text_safe_landscape,
    mask_from_boxes,
    merge_text_boxes,
    natural_background_extend,
    render_full_frame_format,
    restore_protected_pixels,
    save_jpg,
)


def sample_image(size: tuple[int, int] = (420, 420)) -> Image.Image:
    img = Image.new("RGB", size, (180, 150, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, size[0] - 40, size[1] - 40), fill=(80, 120, 170))
    draw.rectangle((110, 170, 310, 230), fill=(245, 245, 245))
    draw.text((130, 190), "TITLE", fill=(0, 0, 0))
    return img


def test_opencv_core_apis_available() -> None:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    assert not cascade.empty()

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    cv2.GaussianBlur(arr, (0, 0), 1.0)


def test_mask_from_boxes() -> None:
    mask = mask_from_boxes((100, 80), [(10, 10, 30, 20)], dilation=3)
    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8
    assert int(mask.max()) == 255


def test_multilang_boxes_are_merged() -> None:
    assert merge_text_boxes([[(10, 10, 40, 30)], [(35, 12, 60, 32)]]) == [(10, 10, 60, 32)]


def test_japanese_ocr_candidate_and_removal_without_model_download(monkeypatch) -> None:
    class FakeReader:
        def readtext(self, _arr, detail=1, paragraph=False):
            return [([(20, 20), (180, 20), (180, 55), (20, 55)], "ふたりが始まる。", 0.18)]

    monkeypatch.setattr("covermorph.processor.easyocr_reader", lambda *_args, **_kwargs: FakeReader())
    img = Image.new("RGB", (240, 100), (170, 170, 170))
    boxes = detect_text_boxes_multilang(img, ("ja", "en"))
    assert boxes
    assert inpaint_text_opencv(img, boxes).size == img.size


def test_opencv_text_removal_keeps_size() -> None:
    img = sample_image()
    out = inpaint_text_opencv(img, [(105, 165, 320, 240)])
    assert out.size == img.size
    assert out.mode == "RGB"


def test_output_sizes() -> None:
    img = sample_image()
    preset = PRESETS["OldPopLounge"]
    assert make_square(img).size == (1400, 1400)
    assert make_text_safe_landscape(img, preset).size == (1920, 1080)
    assert make_shorts(img, preset).size == (1080, 1920)
    assert natural_background_extend(img, (1920, 1080), preset, "thumbnail", anchor="right").size == (1920, 1080)
    assert natural_background_extend(img, (1080, 1920), preset, "shorts").size == (1080, 1920)
    assert make_smart_crop(img, (1920, 1080)).size == (1920, 1080)
    assert make_fit_original(img, (1920, 1080)).size == (1920, 1080)


def test_preview_renderer_is_shared_geometry_for_saved_formats(tmp_path: Path) -> None:
    img = sample_image()
    preset = PRESETS["OldPopLounge"]
    for kind, size in (("thumbnail", (1920, 1080)), ("shorts", (1080, 1920))):
        preview, _ = render_full_frame_format(img, size, preset, kind, mode="natural")
        saved_path = tmp_path / f"{kind}_shared.jpg"
        save_jpg(preview, saved_path)
        with Image.open(saved_path) as saved:
            assert saved.size == preview.size


def test_jpg_save_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "cover.jpg"
    save_jpg(make_square(sample_image()), path)
    with Image.open(path) as reopened:
        assert reopened.size == (1400, 1400)
        assert reopened.mode == "RGB"


def test_protected_pixel_composite() -> None:
    generated = Image.new("RGB", (64, 64), (255, 0, 0))
    protect = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    patch = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
    protect.alpha_composite(patch, (24, 24))
    restored = restore_protected_pixels(generated, protect)
    assert restored.getpixel((30, 30)) == (0, 255, 0)
    assert restored.getpixel((5, 5)) == (255, 0, 0)


def test_outpaint_canvas_masks_match_output_size() -> None:
    img = sample_image()
    canvas, mask, protect = build_outpaint_canvas(img, (1920, 1080), anchor="right")
    assert canvas.size == (1920, 1080)
    assert mask.size == (1920, 1080)
    assert protect.size == (1920, 1080)


def test_full_frame_outpaint_canvas_masks_match_output_size() -> None:
    img = sample_image()
    preset = PRESETS["OldPopLounge"]
    canvas, mask, protect = build_full_frame_outpaint_canvas(img, (1920, 1080), preset, "thumbnail", anchor="right")
    assert canvas.size == (1920, 1080)
    assert mask.size == (1920, 1080)
    assert protect.size == (1920, 1080)


def test_full_frame_local_extension_has_no_black_or_white_borders() -> None:
    img = sample_image()
    preset = PRESETS["OldPopLounge"]
    for out in (
        natural_background_extend(img, (1920, 1080), preset, "thumbnail", anchor="right"),
        natural_background_extend(img, (1080, 1920), preset, "shorts"),
    ):
        arr = np.asarray(out.convert("RGB"))
        edges = np.concatenate(
            [
                arr[:8].reshape(-1, 3),
                arr[-8:].reshape(-1, 3),
                arr[:, :8].reshape(-1, 3),
                arr[:, -8:].reshape(-1, 3),
            ]
        )
        assert not np.all(edges.mean(axis=0) < 5)
        assert not np.all(edges.mean(axis=0) > 250)


def test_full_frame_protected_pixels_are_restored() -> None:
    img = Image.new("RGB", (120, 120), (20, 180, 70))
    preset = PRESETS["OldPopLounge"]
    _canvas, _mask, protect = build_full_frame_outpaint_canvas(
        img,
        (192, 108),
        preset,
        "thumbnail",
        anchor="right",
    )
    generated = Image.new("RGB", (192, 108), (255, 0, 0))
    restored = restore_protected_pixels(generated, protect)
    assert restored.getpixel((150, 54)) != (255, 0, 0)
