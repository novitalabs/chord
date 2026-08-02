"""Kernel timing helpers.

Three methods are available, selected by :func:`benchmark`:

``triton`` (default) calls ``triton.testing.do_bench(warmup=100, rep=1000)``,
byte-for-byte the same call the upstream Humming ``bench_humming.py`` uses, so
numbers here line up directly with its published tables.  It times the wall clock
around the call, so launch overhead is included.

``kineto`` reads the kernel's own GPU duration out of the profiler, excluding CPU
launch overhead.  It reports the kernel's intrinsic speed and reads a few percent
higher than ``triton`` on these shapes; use it to compare kernels rather than to
compare against the Humming tables.

``events`` is a plain CUDA-event loop, kept as the fallback when neither Triton
nor the profiler is available.

Every method flushes L2 first.  An MoE weight tensor is far larger than L2, but
the small shapes fit entirely, and without a flush they would be served from
cache and report a bandwidth the production shapes can never reach.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable
from contextlib import contextmanager

import torch

BenchMethod = str
BENCH_METHODS = ("triton", "kineto", "events")
DEFAULT_BENCH_METHOD = "triton"

# L2 on H200 is 50 MB, so 256 MB is already several times what it takes to evict
# it.  This matches Triton's ``do_bench``, which is what the upstream Humming
# benchmark uses, so numbers here stay comparable with its published tables.
#
# DeepGEMM instead flushes an "excessive" 8 GB (deep_gemm/testing/bench.py), whose
# stated purpose is to give the GPU some chill time between iterations rather than
# to evict L2.  That ~2 ms gap lets clocks recover, so it reports the peak a single
# kernel can hit rather than sustained throughput, and reads 10-15% higher on the
# prefill shapes here.  Set CHORD_FLUSH_L2_BYTES=8000000000 to measure that way.
_DEFAULT_FLUSH_L2_BYTES = int(256e6)
_FLUSH_L2_BYTES = int(
    os.environ.get("CHORD_FLUSH_L2_BYTES", _DEFAULT_FLUSH_L2_BYTES)
)

# The indexed W4A16 kernel is emitted by the ``humming`` template, so this is the
# symbol prefix the profiler reports for it.
KERNEL_NAME = "humming"


@contextmanager
def _suppress_stdout_stderr():
    """Hide the profiler's own chatter without hiding our table."""

    with open(os.devnull, "w") as null_file:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        os.dup2(null_file.fileno(), stdout_fd)
        os.dup2(null_file.fileno(), stderr_fd)
        sys.stdout = sys.stderr = null_file
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)


def flush_l2(device: torch.device) -> None:
    torch.empty(
        _FLUSH_L2_BYTES // 4, dtype=torch.int32, device=device
    ).zero_()


# Match bench_humming.py exactly: triton.testing.do_bench(warmup=100, rep=1000).
_TRITON_WARMUP_MS = 100
_TRITON_REP_MS = 1000


def bench_triton(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup_ms: int = _TRITON_WARMUP_MS,
    rep_ms: int = _TRITON_REP_MS,
) -> tuple[torch.Tensor, float]:
    """Time ``function`` with Triton's ``do_bench``, matching upstream Humming.

    ``do_bench`` clears L2 with its own cache buffer between iterations and sizes
    the warmup/rep counts from the target wall-clock windows, so this is the
    method whose numbers are directly comparable with the Humming tables.
    """

    import triton.testing

    with torch.cuda.device(device):
        output = function()
        torch.cuda.synchronize(device)
        milliseconds = triton.testing.do_bench(
            function, warmup=warmup_ms, rep=rep_ms
        )
    return output, milliseconds / 1e3


def benchmark(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    method: BenchMethod = DEFAULT_BENCH_METHOD,
    iterations: int = 20,
) -> tuple[torch.Tensor, float]:
    """Dispatch to the requested timing method, returning (output, seconds)."""

    if method == "triton":
        return bench_triton(function, device=device)
    if method == "kineto":
        return bench_kineto(function, device=device, iterations=iterations)
    if method == "events":
        return bench_events(function, device=device, iterations=iterations)
    raise ValueError(
        f"unknown benchmark method {method!r}; choose from {BENCH_METHODS}"
    )


def bench_events(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int = 5,
    iterations: int = 10,
) -> tuple[torch.Tensor, float]:
    """Time ``function`` with CUDA events, flushing L2 between iterations.

    Returns the last output and the mean wall time per call in seconds.  This
    includes launch overhead, so :func:`bench_kineto` is preferred for small
    shapes; it is kept as the fallback when profiling is unavailable.
    """

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    with torch.cuda.device(device):
        # The first call may compile the launcher and kernel; keep it out of the
        # timed region just as the production serving path does.
        output = function()
        torch.cuda.synchronize(device)
        for _ in range(warmup):
            output = function()
        torch.cuda.synchronize(device)

        # Time each iteration separately with a flush in between, so no
        # iteration is served from the L2 the previous one warmed.  The flush
        # itself stays outside the summed windows.
        stream = torch.cuda.current_stream(device)
        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iterations)
        ]
        for start, end in events:
            flush_l2(device)
            start.record(stream)
            output = function()
            end.record(stream)
        torch.cuda.synchronize(device)
        seconds = (
            sum(start.elapsed_time(end) for start, end in events)
            / iterations
            / 1e3
        )
    return output, seconds


def bench_kineto(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    kernel_name: str = KERNEL_NAME,
    iterations: int = 20,
    flush: bool = True,
) -> tuple[torch.Tensor, float]:
    """Return the kernel's mean GPU time in seconds, excluding launch overhead.

    Falls back to :func:`bench_events` when the profiler is disabled (the
    NVIDIA tool env var) or when the kernel does not appear in the trace.
    """

    if int(os.environ.get("CHORD_USE_NVIDIA_TOOLS", 0)):
        # Profiling conflicts with Nsight Systems/Compute and Compute Sanitizer.
        return bench_events(function, device=device, iterations=iterations)

    with torch.cuda.device(device):
        output = function()
        torch.cuda.synchronize(device)

        # Initialize kineto/CUPTI before the measured pass so its setup cost
        # does not land inside the reported kernel time.
        if not getattr(bench_kineto, "_kineto_initialized", False):
            with _suppress_stdout_stderr():
                # This pass exists only to pay CUPTI's startup cost; its events
                # are deliberately discarded, so acc_events stays off.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*Profiler clears events.*"
                    )
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA]
                    ):
                        function()
            torch.cuda.synchronize(device)
            bench_kineto._kineto_initialized = True

        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with _suppress_stdout_stderr():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                acc_events=True,
            ) as profiler:
                # Two passes: the first is the profiler's warmup window, the
                # second is the active one it actually records.
                for _ in range(2):
                    for _ in range(iterations):
                        if flush:
                            flush_l2(device)
                        output = function()
                    torch.cuda.synchronize(device)
                    profiler.step()

        lines = (
            profiler.key_averages()
            .table(sort_by="cuda_time_total", max_name_column_width=100)
            .split("\n")
        )

    seconds = _parse_kernel_time(lines, kernel_name)
    if seconds is None:
        return bench_events(function, device=device, iterations=iterations)
    return output, seconds


_TIME_UNITS = {"ms": 1e3, "us": 1e6, "s": 1.0}


def _parse_kernel_time(lines: list[str], kernel_name: str) -> float | None:
    """Average the profiler table's per-kernel time for ``kernel_name``."""

    total_time = 0.0
    total_calls = 0
    for line in lines:
        if kernel_name not in line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        time_field, count_field = fields[-2], fields[-1]
        try:
            calls = int(count_field)
        except ValueError:
            continue
        for unit, scale in _TIME_UNITS.items():
            if time_field.endswith(unit):
                try:
                    value = float(time_field[: -len(unit)])
                except ValueError:
                    break
                total_time += value / scale * calls
                total_calls += calls
                break
    if total_calls == 0:
        return None
    return total_time / total_calls


__all__ = [
    "BENCH_METHODS",
    "DEFAULT_BENCH_METHOD",
    "KERNEL_NAME",
    "benchmark",
    "bench_events",
    "bench_kineto",
    "bench_triton",
    "flush_l2",
]
