from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def package_data(package: str) -> list[str]:
    """Return runtime source files shipped with a JIT-backed package."""
    package_root = ROOT / "chord_kernels" / package
    data: list[str] = []
    for directory in ("include", "csrc"):
        source_root = package_root / directory
        if source_root.exists():
            data.extend(
                path.relative_to(package_root).as_posix()
                for path in source_root.rglob("*")
                if path.is_file()
            )
    for filename in ("SOURCE", "SOURCE.md", "LICENSE"):
        if (package_root / filename).is_file():
            data.append(filename)
    return sorted(data)


setup(
    # ``chord`` is the humming-compatible facade over ``chord_kernels``; it is
    # import-root-only (no kernel sources), so it needs no package_data.
    packages=find_packages(
        include=("chord_kernels", "chord_kernels.*", "chord", "chord.*")
    ),
    # The implementation is packaged under the ``operator`` subpackage.
    package_data={"chord_kernels.operator": package_data("operator")},
    include_package_data=False,
    zip_safe=False,
)
