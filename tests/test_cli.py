"""Smoke tests for the command-line skeleton."""

import subprocess
import sys

import pytest

from sonar_pipeline.__main__ import COMMANDS, HANDLERS


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sonar_pipeline", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_all_commands():
    result = run_cli("--help")
    assert result.returncode == 0
    for name in COMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("command", [c for c in COMMANDS if c not in HANDLERS])
def test_unimplemented_command_fails(command):
    result = run_cli(command)
    assert result.returncode != 0
    assert "not implemented" in result.stderr
