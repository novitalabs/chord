"""Pytest controls for the optional indexed W4A16 performance checks."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="run tests marked perf (disabled by default)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-perf"):
        return
    skip_perf = pytest.mark.skip(
        reason="performance tests are disabled; pass --run-perf to enable"
    )
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)
