"""Shared roofline model and table rendering for the W4A16 benchmark tests.

``test_w4a16_indexed.py`` and ``test_w4a16_grouped.py`` report the same
quantities for different launch conventions, so the arithmetic (device peaks,
roofline ceilings, DRAM traffic, cosine error) and the table rendering live here
while each test module keeps its own case table, fixtures and column layout.
"""

from __future__ import annotations

import torch

# Per-architecture roofline peaks.  The INT4 weight is dequantized to BF16
# before the MMA, so the compute ceiling is the dense BF16 tensor rate rather
# than an INT4 rate.  H200 SXM: 989 TFLOPS BF16, 4.8 TB/s HBM3e.  B200/B300:
# 2250 TFLOPS dense BF16; bandwidth follows the shipping memory clock
# (3996 MHz x 7680-bit = 7.67 TB/s) rather than the 8 TB/s spec-sheet figure.
ROOFLINE_PEAKS: dict[tuple[int, int], tuple[float, float]] = {
    (9, 0): (989.0, 4800.0),
    (10, 0): (2250.0, 7700.0),
    (10, 3): (2250.0, 7700.0),
}

# Width added to whichever throughput column carries the utilization percentage.
UTILIZATION_WIDTH = 8


def device_peaks(device: torch.device | None = None) -> tuple[float, float]:
    """(BF16 TFLOPS, GB/s) peaks for ``device``."""
    capability = torch.cuda.get_device_capability(device)
    peaks = ROOFLINE_PEAKS.get(capability)
    if peaks is None:
        # An unlisted device still benchmarks; fall back to the H200 numbers
        # rather than refusing, since the ceilings only scale the util column.
        peaks = ROOFLINE_PEAKS[(9, 0)]
    return peaks


def roofline(
    flops: int, total_bytes: int, device: torch.device | None = None
) -> tuple[float, float]:
    """Return the (TFLOPS, GB/s) ceilings for this arithmetic intensity.

    The workload sits on one ray of slope ``AI = flops / bytes`` in the roofline
    plane, clipped by the two hardware peaks.  Both ceilings describe the same
    point, so the utilization percentage is identical in either unit; reporting
    both shows which wall is the binding one.
    """
    peak_tflops, peak_gbps = device_peaks(device)
    if total_bytes <= 0:
        return peak_tflops, peak_gbps
    intensity = flops / total_bytes
    roof_tflops = min(peak_tflops, peak_gbps * intensity / 1e3)
    roof_gbps = min(peak_gbps, peak_tflops * 1e3 / intensity)
    return roof_tflops, roof_gbps


def traffic_bytes(
    num_experts: int, n: int, k: int, input_rows: int, output_rows: int
) -> int:
    """DRAM traffic for one W4A16 MoE launch.

    ``num_experts`` counts only the experts whose weight is actually read: with
    a random router and a small token count most experts are never touched, so
    counting all of them would overstate achieved bandwidth.
    """
    weight_bytes = num_experts * n * (k // 2)  # INT4, 2 codes per byte
    scale_bytes = num_experts * n * (k // 32) * 2  # BF16 group-32 scales
    activation_bytes = input_rows * k * 2  # BF16
    output_bytes = output_rows * n * 2  # BF16
    return weight_bytes + scale_bytes + activation_bytes + output_bytes


def error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    """(cosine difference, max absolute difference) between two tensors."""
    actual = actual.float()
    reference = reference.float()
    denominator = (actual.square() + reference.square()).sum()
    cosine_diff = 0.0
    if denominator.item() != 0:
        cosine_diff = float(1 - (2 * (actual * reference).sum() / denominator).item())
    max_abs_diff = float((actual - reference).abs().max().item())
    return cosine_diff, max_abs_diff


def cosine_diff(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Cosine difference alone, for callers that do not need the max abs term."""
    return error_metrics(actual, reference)[0]


def throughput_columns(
    columns: tuple[tuple[str, int], ...], *, compute_bound: bool
) -> tuple[tuple[str, int], ...]:
    """Widen the throughput column that will carry the utilization percentage.

    The percentage rides in the column of the wall that binds -- ``TFLOPS`` for
    compute-bound prefill, ``GB/s`` for bandwidth-bound decode -- so the two
    kinds of block stay the same total width.
    """
    target = "TFLOPS" if compute_bound else "GB/s"
    return tuple(
        (name, size + UTILIZATION_WIDTH) if name == target else (name, size)
        for name, size in columns
    )


def format_throughput(
    tflops: float,
    roof_tflops: float,
    gbps: float,
    roof_gbps: float,
    *,
    compute_bound: bool,
) -> tuple[str, str]:
    """Render the (TFLOPS, GB/s) cells, with the percentage on the binding one."""
    utilization = 0.0
    if compute_bound:
        utilization = 100.0 * tflops / roof_tflops if roof_tflops else 0.0
    else:
        utilization = 100.0 * gbps / roof_gbps if roof_gbps else 0.0
    tflops_cell = f"{tflops:.0f}/{roof_tflops:.0f}"
    gbps_cell = f"{gbps:.0f}/{roof_gbps:.0f}"
    if compute_bound:
        tflops_cell += f" ({utilization:.1f}%)"
    else:
        gbps_cell += f" ({utilization:.1f}%)"
    return tflops_cell, gbps_cell


def print_group(title: str, columns: tuple[tuple[str, int], ...]) -> None:
    """Start a block of rows with its own heading and column header."""
    width = sum(size for _, size in columns) + len(columns) - 1
    print(f"\n{title}", flush=True)
    print("=" * width, flush=True)
    print(" ".join(name.rjust(size) for name, size in columns), flush=True)
    print("-" * width, flush=True)


def print_row(
    values: tuple[str, ...], columns: tuple[tuple[str, int], ...]
) -> None:
    """Print one right-aligned row against ``columns``."""
    print(
        " ".join(value.rjust(size) for value, (_, size) in zip(values, columns)),
        flush=True,
    )
