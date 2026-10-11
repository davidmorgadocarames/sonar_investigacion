"""DT-001: the image works with any UID (compose.yaml runs it as the host user)."""

import os
from pathlib import Path

import pytest


def in_image() -> bool:
    return os.environ.get("MPLCONFIGDIR") == "/tmp/matplotlib"


def test_home_and_matplotlib_cache_are_writable():
    if not in_image():
        pytest.skip("not running in the Docker image")
    home = Path(os.environ["HOME"])
    assert os.access(home, os.W_OK), f"HOME={home} is not writable by UID {os.getuid()}"
    import matplotlib

    cache = Path(matplotlib.get_cachedir())
    assert cache == Path("/tmp/matplotlib") and os.access(cache, os.W_OK)


def test_pytest_cache_is_disabled():
    if not in_image():
        pytest.skip("not running in the Docker image")
    # /app is not writable by a host UID other than 1000: pytest must not try to write there.
    assert "no:cacheprovider" in os.environ.get("PYTEST_ADDOPTS", "")
