# indexed W4A16 的 C++ JIT（SM100 / SM103）

`CHORD_CPP_JIT=1` 时，indexed W4A16 GEMM（humming kernel，MMA / swap-AB MMA，可带 stream-K）和 repack_w4a16 每次调用时的选配置、拼源码、缓存、编译（进程内调 NVRTC）、加载、启动都在 C++ 里做。默认 0：走原来的 Python JIT，行为不变。

## 1. 生效条件

同时满足才走 C++ JIT，否则照旧走 Python JIT：

- `CHORD_CPP_JIT=1`；
- 张量在 CUDA 上，设备是 SM100 或 SM103；
- GEMM：权重是 MMA 排法（WGMMA 只有 sm_90a 有，不在 C++ JIT 里）；repack 两种排法都支持。

SM90（WGMMA、DeepGEMM grouped）、H200 的配置规则都不经过 C++ JIT。

## 2. 调用链

```
dispatch.forward_w4a16（层 / vLLM 走这里）
  └─ cpp_jit.select_layer_config → chord::indexed::select_config            选配置
  └─ api.w4a16_indexed（原有的参数检查）
       └─ cpp_jit.get_indexed_kernel（lru_cache）→ chord::indexed::register_gemm  拼源码、编译 / 读缓存、加载、注册，返回 kernel id
       └─ cpp_jit.indexed_gemm → torch.ops.chord_cpp.indexed_gemm（检查张量、分配输出、当前 stream）
            └─ chord::indexed::gemm                                          stream-K 锁、启动

packing.pack_w4a16
  └─ cpp_jit.repack_w4a16 → torch.ops.chord_cpp.repack_w4a16 → chord::indexed::repack_w4a16
```

`api.w4a16_indexed(config=None)` 的默认配置在同样条件下也由 C++ 算（`cpp_jit.default_config`）。

## 3. 文件

| 文件 | 内容 |
|---|---|
| `csrc/jit/jit.{h,cpp}` | 通用 JIT：进程内 NVRTC、盘上缓存、加载、读 cubin 常量、进度条、预编译、PCH。只依赖 CUDA 驱动 API 和 NVRTC |
| `csrc/indexed/humming.{h,cpp}` | GEMM：配置检查、拼源码（含 MMA 指令）、注册、启动、stream-K 锁 |
| `csrc/indexed/repack.{h,cpp}` | repack：拼源码、启动 |
| `csrc/indexed/select.{h,cpp}` | 选配置：`blackwell_decode_ep8` 的规则、profile 默认配置、按 m 查调优表、routing block-M 覆盖、算子层默认配置 |
| `csrc/indexed/torch_ops.cpp` | torch op 和 pybind 函数 |
| `cpp_jit.py` | 编扩展（第一次用时，`torch.utils.cpp_extension`）、转发、`precompile_indexed`、`layer_kernels` |

`csrc/` 指 `chord_kernels/operator/csrc/`。纯 C++ 那层（`chord::jit`、`chord::indexed`）不依赖 torch 和 Python。

## 4. Python、C++ 各有一份的东西

开关关闭时、SM90 上、给 vLLM 发布调优表时还用 Python 那份，所以下面这些两边各有一份，改一份要改另一份。C++ 那份的文件开头都写了对应的 Python 位置；Python 那边只有选配置的几处（`tuning.py`、`api.py`、`dispatch.py`）有注释指向 C++，kernel 模板、MMA 指令、repack、launcher 那几处没有：

| 内容 | Python | C++ |
|---|---|---|
| kernel 源码模板 | `kernel/humming.py` `CODE_TEMPLATE` | `humming.cpp` `kernel_source` |
| MMA 指令文本 | `config/mma.py` `_MmaOp` | `humming.cpp` `mma_op_class` |
| 配置检查 | `HummingKernel._validate_config`、`_select_mma_op`、`api._get_indexed_kernel`、`IndexedKernelConfig` | `humming.cpp` `check_config` |
| 注册、启动参数、stream-K 锁 | `csrc/launcher/launcher.cpp` `register_kernel` / `launch_kernel` | `humming.cpp` |
| repack 源码和启动 | `kernel/repack_weight.py` | `repack.cpp` |
| Blackwell 规则、默认配置 | `tuning.py` `_select_indexed_kernel_config`，`api._compatible_block_n` / `IndexedKernelConfig.default` | `select.cpp` |
| 调优表解析、按 m 查行、block-M 覆盖 | `tuning._resolve_tuning_config`、`dispatch._config_with_block_m` | `cpp_jit._table_id`（解析）、`select.cpp`（查行、覆盖） |

仓库里没有检查两份一致的测试。改了上面任一处，另一份要跟着改，再按第 7 节验证。

## 5. 缓存

- 位置 `$CHORD_CACHE_DIR/cpp/indexed_gemm/<哈希>/`、`$CHORD_CACHE_DIR/cpp/repack/<哈希>/`，和 Python JIT 的缓存分开放；`CHORD_CACHE_DIR` 的默认值和 Python JIT 一样。
- 键：NVRTC 版本、编译选项（含架构和 `-I`）、kernel 实例化表达式、内存头文件、源码、`include/` 下所有 `.cuh` 的修改时间。改了 `.cuh` 就重编。
- 目录里：`key.txt`、`kernel.cu`、`nvrtc.log`、`name.txt`、`kernel.cubin`、`lock`。`kernel.cubin` 最后写，看到它就说明其余文件已写完。
- 多进程：没有 `kernel.cubin` 时对 `lock` 加 `flock`，拿到锁后再看一次，还没有才编。进程内每个 (设备, 程序) 只编（或读）一次、只加载一次。
- 编译选项、头文件、源码和 Python JIT 一样，两种 JIT 编出的 cubin 逐字节相同。

## 6. 预编译头（PCH）

- 真要编译时（缓存没命中）加 NVRTC `-pch`（CUDA 12.8+）。chord 的 kernel 源码第一行都是 include 同一个 kernel 头文件，同一个进程里第一次编译生成 PCH，之后的编译跳过解析这个头文件。B300 上冷启动预编译 10 个 kernel：不加 5.0 s，加了 0.9 s。
- `-pch` 不改变 cubin，不进缓存键。
- 只在同一个进程里有用：PCH 只能由生成它的 NVRTC 库实例使用（PCH 堆地址每个进程不同；别的进程用 `--use-pch` 读，NVRTC 报警告 #20282-D 不用）。不指定 `--pch-dir`，NVRTC 用每个进程一个的临时目录（`$TMPDIR/__nvrtc_auto_pch*`），卸载时删掉。
- NVRTC 复用 PCH 时不检查头文件的修改时间，所以进程里 `.cuh` 一变，这个进程后面就不再加 `-pch`。
- 内存：PCH 堆默认 256 MB，只占虚拟地址，编完把物理内存还给系统。B300 上编 10 个 kernel，峰值常驻内存和不加时差不多（约 +77 MB），编完常驻内存 +17 MB（不加时 +78 MB）。
- `CHORD_JIT_PCH=0` 关掉。

## 7. 验证

现有测试在开关打开时走 C++ 路径，在 SM100 / SM103 上跑：

```bash
CHORD_CPP_JIT=1 python -m pytest -m gpu tests/          # 含 indexed 层的端到端正确性
CHORD_CPP_JIT=1 python tests/test_w4a16_indexed.py      # Blackwell 用例的正确性 + 性能表
```

这些只检查结果对不对（和参考结果比 cos），不检查 C++ 和 Python 两份是否一致。要确认一致，比对：

- 选配置：同一组输入（`blackwell_decode_ep8` 两个形状扫 routed-M、表外形状、调优表、block-M 覆盖）两边选出的 `IndexedKernelConfig`、报错类型和提示文字都相同；
- kernel：两种 JIT 的缓存目录里，同一个配置的 `kernel.cu`、kernel 名字、`kernel.cubin` 逐字节相同；
- 输出：两条路径的 GEMM 输出逐位相同。stream-K 默认在 FP32 workspace 里归约，结果是确定的；只有显式 `stream_k_fp32=False` 的 BF16 归约每次跑会有很小的差别。

## 8. 不支持的、没验证的

- SM90 / WGMMA：不在 C++ JIT 里。
- 调优表：只有 JSON 字符串（vLLM 的用法）由 C++ 查行；Python 的 dict / list 照旧在 Python 里处理。字符串在第一次用到时整张解析，表里有格式不对的行会在第一次调用时报错（Python JIT 是 m 落到那一行时才报错）。
- 录 CUDA graph：调用路径不加主机同步；stream-K 锁和 launcher 一样，在 relaxed 录图模式下分配、在私有 stream 上清零，所以某个 stream 的第一次 stream-K 启动可以发生在录图中（`test_stream_k_first_launch_inside_cuda_graph_capture`）。
- 从 C++ 直接调用（不经过 Python）：会编、会链，没有实跑。
