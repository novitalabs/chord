# Derived from inclusionAI/humming; modified for chord_kernels.
# Provenance and the list of changes are in chord_kernels/operator/SOURCE.md.

import textwrap


_BF16_BITS = 16
_FP32_BITS = 32


def _register_count(rows: int, cols: int, bits: int) -> int:
    total_bits = rows * cols * bits
    if total_bits % (32 * 32):
        raise ValueError(
            f"MMA fragment {rows}x{cols}x{bits} does not fill whole registers"
        )
    return total_bits // (32 * 32)


def _format_asm(code: str, indent: int) -> str:
    code = textwrap.dedent(code).strip()
    prefix = " " * indent
    return "".join(f"\n{prefix}{line}" for line in code.splitlines())


class _MmaOp:
    def __init__(self, m: int, n: int, k: int) -> None:
        self.shape = (m, n, k)
        self.reg_a_count = _register_count(m, k, _BF16_BITS)
        self.reg_b_count = _register_count(k, n, _BF16_BITS)
        self.reg_c_count = _register_count(m, n, _FP32_BITS)

    def to_cpp_str(self) -> str:
        m, n, k = self.shape
        lines = [
            "static constexpr bool kUseWgmma = false;",
            f"using MmaShape = Shape<{m}, {n}, {k}>;",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = float[{self.reg_c_count}];",
            "",
            "CUDA_INLINE",
            "static void fma(uint32_t *a, uint32_t *b, float *c, float *d) {",
            *self._generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]
        return "\n".join("  " + line if line else line for line in lines)

    def _generate_ptx(self, indent: int) -> str:
        m, n, k = self.shape
        counts = (
            self.reg_c_count,
            self.reg_a_count,
            self.reg_b_count,
            self.reg_c_count,
        )
        placeholders = []
        start = 0
        for count in counts:
            placeholders.append(
                "{" + ", ".join(f"%{i}" for i in range(start, start + count)) + "}"
            )
            start += count

        a_params = ", ".join(f' "r"(a[{i}])' for i in range(self.reg_a_count))
        b_params = ", ".join(f' "r"(b[{i}])' for i in range(self.reg_b_count))
        c_params = ", ".join(f' "f"(c[{i}])' for i in range(self.reg_c_count))
        d_params = ", ".join(f'"+f"(d[{i}])' for i in range(self.reg_c_count))
        asm_op = f"mma.sync.aligned.m{m}n{n}k{k}.row.col.f32.bf16.bf16.f32"

        return _format_asm(
            f"""
            asm volatile(
              "{asm_op} "
              "{', '.join(placeholders)};\\n"
              : {d_params}
              : {a_params},
                {b_params},
                {c_params}
            );
            """,
            indent,
        )


class _WgmmaOp:
    def __init__(self, m: int, n: int, k: int) -> None:
        self.shape = (m, n, k)
        # WGMMA fragments are distributed across one four-warp group.
        self.reg_b_count = _register_count(n, k, _BF16_BITS) // 4
        self.reg_c_count = _register_count(m, n, _FP32_BITS) // 4

    def to_cpp_str(self) -> str:
        m, n, k = self.shape
        lines = [
            "static constexpr bool kUseWgmma = true;",
            f"using MmaShape = Shape<{m}, {n}, {k}>;",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = float[{self.reg_c_count}];",
            "",
            "CUDA_INLINE",
            "static void fma(uint64_t &desc, uint32_t *b, float *d, bool pred = true) {",
            *self._generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]
        return "\n".join("  " + line if line else line for line in lines)

    def _generate_ptx(self, indent: int) -> str:
        m, n, k = self.shape
        output_placeholders = "{" + ", ".join(
            f"%{i}" for i in range(self.reg_c_count)
        ) + "}"
        b_start = self.reg_c_count
        b_placeholders = "{" + ", ".join(
            f"%{i}" for i in range(b_start, b_start + self.reg_b_count)
        ) + "}"
        desc_placeholder = self.reg_c_count + self.reg_b_count
        pred_placeholder = desc_placeholder + 1

        b_params = ", ".join(f' "r"(b[{i}])' for i in range(self.reg_b_count))
        cd_params = [f'"+f"(d[{i}])' for i in range(self.reg_c_count)]
        cd_lines = []
        for offset in range(0, len(cd_params), 4):
            cd_lines.append(", ".join(cd_params[offset : offset + 4]))
        cd_param_str = ",\n    ".join(cd_lines)

        # Project A's shared-memory descriptor occupies WGMMA operand B;
        # dequantized project B registers occupy operand A.
        asm_op = f"wgmma.mma_async.sync.aligned.m{n}n{m}k{k}.f32.bf16.bf16"
        return _format_asm(
            f"""
            asm volatile(
              "{{\\n"
                ".reg .pred p;\\n"
                "setp.ne.b32 p, %{pred_placeholder}, 0;\\n"
                "{asm_op} {output_placeholders}, {b_placeholders}, "
                "%{desc_placeholder}, p, 1, 1, 0;\\n"
              "}}\\n"
              : {cd_param_str}
              : {b_params},
                "l"(desc), "r"((uint32_t)pred)
            );
            """,
            indent,
        )


class MmaOpClass:
    """Generate the fixed BF16-input, FP32-accumulator MMA instruction."""

    @classmethod
    def from_config(
        cls, mma_type: str, m: int, n: int, k: int
    ) -> _MmaOp | _WgmmaOp:
        if not isinstance(mma_type, str):
            raise TypeError("mma_type must be 'mma' or 'wgmma'")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (m, n, k)
        ):
            raise ValueError("MMA shape must contain three positive integers")

        mma_type = mma_type.lower()
        if mma_type == "mma":
            return _MmaOp(m, n, k)
        if mma_type == "wgmma":
            return _WgmmaOp(m, n, k)
        raise ValueError(f"mma_type must be 'mma' or 'wgmma', got {mma_type!r}")
