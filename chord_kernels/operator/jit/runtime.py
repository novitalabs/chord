# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import dataclasses
import threading
from typing import Any, ClassVar

import cuda.bindings.driver as cbd
import torch

from chord_kernels.operator.utils import jit as jit_utils
from chord_kernels.operator.jit.compiler import NVRTCCompiler


@dataclasses.dataclass(kw_only=True)
class KernelRuntime:
    _instances: ClassVar[dict[tuple[str, tuple[Any, ...]], "KernelRuntime"]] = {}

    def __new__(cls, *args, **kwargs):
        def get_value(value):
            if isinstance(value, list):
                value = tuple(value)
            return value

        args_items = tuple(get_value(x) for x in args)
        kwargs_items = tuple((key, get_value(kwargs[key])) for key in sorted(kwargs.keys()))
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else -1
        signature = (
            cls.__name__,
            (("cuda_device", device_index),) + args_items + kwargs_items,
        )

        if signature not in cls._instances or not cls._instances[signature].inited:
            instance = super().__new__(cls)
            cls._instances[signature] = instance
            instance.inited = False
            instance.cubin_loaded = False
        return cls._instances[signature]

    def __post_init__(self):
        if self.inited:
            return
        self.init_sm_version()
        self.init_kernel()
        self.inited = True

    def init_kernel(self):
        raise NotImplementedError

    def init_sm_version(self):
        device_props = torch.cuda.get_device_properties()
        sm_version = device_props.major * 10 + device_props.minor
        self.sm_version = sm_version
        self.sm_version_str = str(sm_version)
        if self.sm_version >= 90:
            self.sm_version_str += "a"

    @staticmethod
    def _ensure_cuda_context():
        torch.cuda.set_device(torch.cuda.current_device())

    def prepare(self):
        self._ensure_cuda_context()
        kernel_expr = getattr(self, "kernel_expr", None)
        kernel_filename = NVRTCCompiler.compile(
            self.code,
            sm_version=self.sm_version_str,
            kernel_expr=kernel_expr,
        )
        kernel_name = jit_utils.find_kernel_name_in_cubin(kernel_filename, self.name)
        self.kernel_name = kernel_name
        self.kernel_filename = kernel_filename
        if threading.current_thread() is threading.main_thread():
            self.load_cubin()

    def load_cubin(self):
        if self.cubin_loaded:
            return None
        kernel_filename = self.kernel_filename
        kernel_name = self.kernel_name
        result, lib = cbd.cuLibraryLoadFromFile(kernel_filename.encode(), [], [], 0, [], [], 0)
        assert result == 0, repr(result)
        result, kernel = cbd.cuLibraryGetKernel(lib, kernel_name.encode())
        assert result == 0, repr(result)
        result, func = cbd.cuKernelGetFunction(kernel)
        assert result == 0, repr(result)
        self.func = func
        self.cubin_loaded = True

    def check_context(self):
        assert threading.current_thread() is threading.main_thread()
        if not self.cubin_loaded:
            self.load_cubin()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
