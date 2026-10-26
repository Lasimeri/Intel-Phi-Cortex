# Stage 0c: F32 transcendental accuracy on the 7120A (2026-10-25 09:03)

Commands:
- `build/host/isa-host gen data/isa` (host: inputs, xorshift64* seed 0x9E3779B97F4A7C15).
- Per set: `phi put` the .in, `phi run /tmp/cortex/isa OP in out` (card: vector instructions only, card/vpu.h), `phi get` the .out.
- `build/host/isa-host check data/isa` (host: libm double references).

Raw: `results/s00c-isa.log`; data `data/isa/*.in|*.out` (2^20 float32 each).

Code check: `llvm-objdump -d --mcpu=knc build/obj/isa-knc.o` shows `vcvtfxpntps2dq $0x50` (round to nearest, exponent adjustment 2^24 = Q8.24) followed by `vexp223ps`. The builtin's first immediate is the imm8; its second (sss) is 0.

| Set | n | Max | Median | p99.9 | Manual bound | Over bound |
|---|---|---|---|---|---|---|
| exp2 vs 2^(x rounded to Q8.24): the instruction | 1,048,560 | 0.9366 ULP | 0.2504 | 0.7486 | 0.99 ULP | 0 |
| exp2 vs exact 2^x (convert + exp223) | 1,048,560 | 1.165 ULP | 0.2529 | 0.8723 | (none) | 90 above 1 ULP |
| rcp23 vs 1/x, \|x\| in [2^-20, 2^20] | 1,048,576 | 0.8991 ULP | 0.2505 | 0.7329 | 0.912 ULP | 0 |
| rsqrt23 vs 1/sqrt(x) | 1,048,576 | 0.704 ULP | 0.2499 | 0.6218 | 0.775 ULP | 0 |
| log2 outside [0.5, 2] | 996,344 | 1.038 ULP | 0.2504 | 0.7279 | 4 ULP | 0 |
| log2 inside [0.5, 2], absolute | 52,232 | 9.266e-08 | 1.669e-08 | 8.232e-08 | 2^-21 = 4.768e-07 | 0 |
| log2 dense [0.5, 2], absolute | 1,048,576 | 9.479e-08 | 1.619e-08 | 7.782e-08 | 4.768e-07 | 0 |

Special values (exp2): 2^-inf = 0, 2^-0 = 2^0 = 1, 2^NaN = 1, 2^1 = 2, 2^2 = 4, 2^-1 = 0.5, 2^-2 = 0.25, 2^10 = 1024, 2^-10 = 0.000976562, 2^-126 = 1.17549e-38, 2^127 = 1.70141e+38, 2^-200 = 0 (saturates), 2^200 = inf, 2^0.5 = 1.41421, 2^-0.5 = 0.707107. Matches the manual's table.

Result: PASS. Every instruction is inside Intel's stated bound on this card. End to end, the F32 exp2 has max 1.165 ULP. Its excess over the instruction's 0.94 comes from rounding x to 2^-24 when |x| < 0.5, as predicted: <= 0.35 ULP extra.

Softmax, sigmoid, SiLU, RMSNorm, division and Adam can stay on the VPU at F32 with no Newton steps.

Also shown: host binaries run from /mnt (ntfs3 exec works).
