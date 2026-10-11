"""D-017 and D-019: the image has exactly the pinned versions, and PyTorch and scikit-learn work."""

import re
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)$")


def pins(name):
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{path} not found (the Dockerfile copies it to /app)")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PIN.match(line).groups() for line in lines if PIN.match(line)]


@pytest.mark.parametrize("name", ["requirements.txt", "requirements-torch.txt"])
def test_installed_versions_match_pins(name):
    for package, pinned in pins(name):
        installed = version(package)
        # torch carries a local label for its build: 2.14.1+cpu or 2.14.1+cu130.
        assert installed.split("+")[0] == pinned, f"{package}: {installed} != {pinned}"


def test_torch_runs_on_cpu():
    import torch

    torch.manual_seed(0)
    x = torch.randn(4, 3)
    layer = torch.nn.Linear(3, 2)
    assert layer(x).shape == (4, 2)


def test_scikit_learn_fits():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array([0, 0, 1, 1])
    assert list(LogisticRegression().fit(x, y).predict(x)) == [0, 0, 1, 1]
