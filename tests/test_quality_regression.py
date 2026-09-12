from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from covermorph.processor import (
    inpaint_text_opencv, make_smart_crop, _paste_axis_extensions,
    render_full_frame_format,
)
from covermorph.pipeline import PipelineOptions, _outpaint_or_fallback
from covermorph.presets import PRESETS


def test_large_title_is_not_smeared_by_classical_inpainting():
    image = Image.new('RGB', (400, 400), 'white')
    with pytest.raises(ValueError, match='LaMa'):
        inpaint_text_opencv(image, [(30, 30, 190, 300)])


def test_crop_preserves_pixels_at_unit_scale():
    pixels = np.random.default_rng(4).integers(0, 256, (192, 192, 3), dtype=np.uint8)
    result = make_smart_crop(Image.fromarray(pixels), (192, 108))
    np.testing.assert_array_equal(np.asarray(result), pixels[42:150])


def test_extension_reflects_without_stretching():
    pixels = np.random.default_rng(7).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    base = Image.new('RGB', (100, 100))
    _paste_axis_extensions(base, Image.fromarray(pixels), 30, 30)
    expected = np.pad(pixels, ((30, 30), (30, 30), (0, 0)), mode='reflect')
    np.testing.assert_array_equal(np.asarray(base), expected)


def test_ai_preview_does_not_masquerade_as_generation():
    with pytest.raises(RuntimeError, match='AI'):
        render_full_frame_format(Image.new('RGB', (100, 100)), (192, 108),
                                 PRESETS['OldPopLounge'], 'thumbnail', mode='ai_natural')


def test_missing_ai_does_not_save_stretched_fallback(tmp_path):
    ai = SimpleNamespace(sdxl_available=lambda: False)
    with pytest.raises(RuntimeError, match='SDXL'):
        _outpaint_or_fallback(tmp_path, ai, Image.new('RGB', (100, 100)),
                             PRESETS['OldPopLounge'], PipelineOptions(output_dir=tmp_path),
                             'shorts', (108, 192), 'center')


def test_lama_merge_preserves_every_unmasked_pixel():
    from covermorph.ai_plugins import AIBackends
    source = Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    mask = Image.new('L', source.size)
    mask.paste(255, (20, 20, 40, 40))
    result = AIBackends.merge_restoration(source, Image.new('RGB', source.size, 'red'), mask)
    outside = np.asarray(mask) == 0
    np.testing.assert_array_equal(np.asarray(source)[outside], np.asarray(result)[outside])


def test_protection_rejects_geometry_mismatch():
    from covermorph.processor import restore_protected_pixels
    with pytest.raises(ValueError):
        restore_protected_pixels(Image.new('RGB', (20, 30)), Image.new('RGBA', (30, 20)))
