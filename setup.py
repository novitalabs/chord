import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

try:  # PEP 660 editable installs; absent on older setuptools.
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:  # pragma: no cover
    editable_wheel = None

ROOT = Path(__file__).resolve().parent
OPERATOR = ROOT / "chord_kernels" / "operator"
CUTLASS_INCLUDE = ROOT / "third_party" / "cutlass" / "include"
CUTLASS_LICENSE = ROOT / "third_party" / "cutlass" / "LICENSE.txt"

sys.path.insert(0, str(ROOT / "scripts"))
from cutlass_closure import ROOTS, compute_closure  # noqa: E402

_SUBMODULE_HINT = (
    f"CUTLASS headers are missing from {CUTLASS_INCLUDE.relative_to(ROOT)}.\n"
    "The NVIDIA CUTLASS sources ship as a git submodule; fetch them with:\n"
    "    git submodule update --init --recursive"
)


def _closure_seeds() -> list[Path]:
    """Kernel sources whose CUTLASS/CuTe includes define the packaged closure."""
    return [OPERATOR / "include" / "deep_gemm", OPERATOR / "include" / "humming", OPERATOR / "csrc"]


def cutlass_closure() -> set[str]:
    if not CUTLASS_INCLUDE.is_dir():
        raise SystemExit(_SUBMODULE_HINT)
    closure = compute_closure(_closure_seeds(), CUTLASS_INCLUDE)
    if not closure:
        raise SystemExit(_SUBMODULE_HINT)
    return closure


def stage_cutlass(target_include: Path) -> None:
    """Copy the CUTLASS/CuTe closure under ``target_include`` beside the owned headers."""
    for rel in sorted(cutlass_closure()):
        destination = target_include / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CUTLASS_INCLUDE / rel, destination)
    # BSD-3-Clause requires the license text to travel with the redistributed source.
    if CUTLASS_LICENSE.is_file():
        for name in ROOTS:
            root_dir = target_include / name
            if root_dir.is_dir():
                shutil.copy2(CUTLASS_LICENSE, root_dir / "LICENSE.txt")


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
    if package == "operator":
        # Staged from the submodule into build_lib, so they are absent from the
        # source tree during a clean wheel build and must be declared directly.
        data.extend(f"include/{rel}" for rel in cutlass_closure())
        data.extend(f"include/{name}/LICENSE.txt" for name in ROOTS)
    return sorted(set(data))


class BuildPy(build_py):
    """Stage the CUTLASS closure into the build tree so it lands in the wheel."""

    def run(self) -> None:
        super().run()
        stage_cutlass(Path(self.build_lib) / "chord_kernels" / "operator" / "include")


class Develop(develop):
    """Mirror the CUTLASS closure into the source tree for legacy editable installs."""

    def run(self) -> None:
        stage_cutlass(OPERATOR / "include")
        super().run()


cmdclass = {"build_py": BuildPy, "develop": Develop}

if editable_wheel is not None:

    class EditableWheel(editable_wheel):
        """Mirror the CUTLASS closure into the source tree for PEP 660 editable installs.

        An editable install maps the package back to this working tree, so the
        JIT include root (resolved relative to the package directory) must carry
        the CUTLASS headers beside the owned ones. These paths are gitignored.
        """

        def run(self) -> None:
            stage_cutlass(OPERATOR / "include")
            super().run()

    cmdclass["editable_wheel"] = EditableWheel

setup(
    # ``chord`` is the humming-compatible facade over ``chord_kernels``, and
    # ``humming`` is the import root frameworks that hardcode the upstream
    # package name resolve (notably vLLM's lazy facade); both are
    # import-root-only (no kernel sources), so they need no package_data.
    packages=find_packages(
        include=(
            "chord_kernels",
            "chord_kernels.*",
            "chord",
            "chord.*",
            "humming",
            "humming.*",
        )
    ),
    # The implementation is packaged under the ``operator`` subpackage.
    package_data={"chord_kernels.operator": package_data("operator")},
    include_package_data=False,
    zip_safe=False,
    cmdclass=cmdclass,
)
