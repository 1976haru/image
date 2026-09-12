from __future__ import annotations

import pytest


@pytest.mark.optional_ai
def test_optional_sdxl_lama_realesrgan_smoke() -> None:
    pytest.skip("Requires locally installed SDXL, LaMa, and Real-ESRGAN assets.")
