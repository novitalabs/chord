# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import glob
import json
import os
import subprocess
from pathlib import Path

from cuda.bindings import nvrtc
from filelock import FileLock

from chord_kernels.operator.utils import jit as jit_utils
from chord_kernels.operator.utils.cuda import filter_cuda_paths
from chord_kernels.operator.utils.nvrtc import may_build_nvrtc_compile_binary


class Compiler:
    @classmethod
    def signature(cls):
        raise NotImplementedError

    @staticmethod
    def operator_include_dir():
        dirname = os.path.dirname(__file__)
        dirname = os.path.abspath(dirname + "/../include/")
        return dirname

    @staticmethod
    def include_dirs():
        return [Compiler.operator_include_dir()]

    @staticmethod
    def cuh_last_update_time():
        dirname = Compiler.operator_include_dir()
        data = {}
        for filename in sorted(glob.glob(f"{dirname}/**/*.cuh", recursive=True)):
            data[filename] = os.stat(filename).st_mtime
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def compile(cls, code, sm_version, kernel_expr):
        flags = cls.get_flags(sm_version)
        signature = f"{cls.__name__}$${cls.signature()}$${flags}$${kernel_expr}$${code}"
        signature += "$$" + Compiler.cuh_last_update_time()
        hash_hex = jit_utils.hash_to_hex(signature)

        cache_dirname = Path(os.path.join(jit_utils.get_chord_cache_dir(), hash_hex))
        cache_filename = cache_dirname / "kernel.cubin"
        cache_dirname.mkdir(exist_ok=True, parents=True)

        lock_filename = jit_utils.get_chord_lock_filename(hash_hex)
        with FileLock(lock_filename):
            if cache_filename.exists():
                return cache_filename.as_posix()

            cache_dirname.mkdir(exist_ok=True, parents=True)
            source_path = os.path.join(cache_dirname, "kernel.cu")
            temporary_cubin = cache_dirname / "kernel_tmp.cubin"
            temporary_cubin.unlink(missing_ok=True)
            try:
                with open(cache_dirname / "kernel.cu", "w") as f:
                    f.write(code)
                with open(cache_dirname / "signature.txt", "w") as f:
                    f.write(signature)

                compile_res = cls._compile(
                    source_path, cache_dirname, sm_version, kernel_expr, flags
                )
                returncode, stdout, stderr = compile_res

                with open(cache_dirname / "stdout.log", "w") as f:
                    f.write(stdout)
                with open(cache_dirname / "stderr.log", "w") as f:
                    f.write(stderr)

                if returncode != 0:
                    print(stderr, flush=True)
                    raise RuntimeError(f"{cls} run failed")
                if not temporary_cubin.is_file():
                    raise RuntimeError(
                        f"{cls} reported success without producing {temporary_cubin}"
                    )

                # Publishing while holding the same lock ensures a waiting process
                # observes either no cubin or the complete final cubin, never a
                # shared temporary file from another compiler invocation.
                os.replace(temporary_cubin, cache_filename)
            except BaseException:
                temporary_cubin.unlink(missing_ok=True)
                raise

            return cache_filename.as_posix()

    @classmethod
    def get_flags(cls, sm_version):
        raise NotImplementedError

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr, flags):
        raise NotImplementedError


class NVRTCCompiler(Compiler):
    _STD_HEADER_SHIMS: dict[str, str] = {
        "climits": "#include <cuda/std/climits>",
        "cfloat": "#include <cuda/std/cfloat>",
        "cstddef": """
            #include <cuda/std/cstddef>
            using namespace cuda::std;
        """,
        "cstdint": """
            #include <cuda/std/cstdint>
            #include <cuda/std/type_traits>
            using namespace cuda::std;
            namespace std {
            using cuda::std::is_same;
            using cuda::std::conditional_t;
            using cuda::std::conditional;
            using cuda::std::enable_if;
            using cuda::std::enable_if_t;
            }
        """,
        "type_traits": """
            #include <cuda/std/type_traits>
            namespace std {
            using cuda::std::is_same;
            using cuda::std::conditional_t;
            using cuda::std::conditional;
            using cuda::std::enable_if;
            using cuda::std::enable_if_t;
            }
        """,
    }

    @classmethod
    def signature(cls):
        _, major, minor = nvrtc.nvrtcVersion()
        return f"nvrtc+{major}.{minor}"

    @classmethod
    def get_flags(cls, sm_version):
        flags = [
            f"--gpu-architecture=sm_{sm_version}",
            "-std=c++17",
            "--use_fast_math",
            "--dopt=on",
            "-extra-device-vectorization",
            "--ptxas-options=-O3",
            "--ptxas-options=--register-usage-level=10",
            "--diag-suppress=39,161,174,177,940",
            "-default-device",
        ]
        for d in cls._get_include_dirs():
            flags.append(f"-I{d}")
        if os.environ.get("CHORD_LINEINFO", "0") == "1":
            flags.append("-lineinfo")
        return flags

    @classmethod
    def _get_include_dirs(cls):
        env = filter_cuda_paths(required_headers=["cuda_runtime.h"])
        return list(cls.include_dirs()) + list(env["include_paths"])

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr, flags):
        binary_path = may_build_nvrtc_compile_binary()

        shims_dir = Path(cache_dirname) / "shims"
        shims_dir.mkdir(exist_ok=True)
        header_args = []
        for header, content in cls._STD_HEADER_SHIMS.items():
            shim_file = shims_dir / header
            with open(shim_file, "w") as f:
                f.write(content)
            header_args += ["--header", f"{header}={shim_file.as_posix()}"]

        target_path = (Path(cache_dirname) / "kernel_tmp.cubin").as_posix()
        cmd = [
            binary_path,
            "--input", source_path,
            "--output", target_path,
            *header_args,
        ]
        if kernel_expr:
            name_expr = " ".join(kernel_expr.split())
            cmd += ["--name-expression", name_expr]
        cmd += ["--", *flags]

        with open(Path(cache_dirname) / "cmdline.json", "w") as f:
            json.dump(cmd, f, ensure_ascii=False)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode, result.stdout, result.stderr
