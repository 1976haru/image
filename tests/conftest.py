from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_v2_engines(request, monkeypatch):
    """Keep tests hermetic: a machine with real Z-Image/FLUX.2 files must not route bridge tests to them.

    Tests that exercise the V2 route opt in with ``@pytest.mark.v2_engines`` and fake the backends.
    """
    if request.node.get_closest_marker("v2_engines"):
        return
    from covermorph import thumbnail_bridge_runtime as runtime

    monkeypatch.setattr(runtime, "_v2_ready", lambda models_dir: {name: False for name in runtime.V2_ENGINES})


def pytest_configure(config):
    config.addinivalue_line("markers", "v2_engines: test runs the Quality Engine V2 bridge route with fakes")
