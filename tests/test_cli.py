"""Smoke tests for the command-line skeleton."""

import re
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


def listed_commands(help_text: str) -> dict[str, str]:
    """Parse the argparse subcommand list: lines like '    qc    Run quality control...'."""
    listed = {}
    for line in help_text.splitlines():
        match = re.fullmatch(r" {4}(\S+)\s{2,}(.+)", line)
        if match:
            listed[match.group(1)] = match.group(2).strip()
    return listed


def test_help_lists_exactly_the_commands():
    result = run_cli("--help")
    assert result.returncode == 0
    listed = listed_commands(result.stdout)
    assert listed == {name: help_text for name, (help_text, _) in COMMANDS.items()}


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_has_its_own_help(command):
    result = run_cli(command, "--help")
    assert result.returncode == 0
    assert result.stdout.startswith(f"usage: sonar_pipeline {command}")


def test_every_command_is_implemented():
    assert set(HANDLERS) == set(COMMANDS)
