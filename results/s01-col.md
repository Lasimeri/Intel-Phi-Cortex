# Stage 1: one column forward on the VPU (2026-10-25 22:5x-23:2x, 7120A)

Code: `lib/cx-col.h` (shapes, layout, splitmix64 init shared by card and host), `card/col.h` (the kernel), `tools/col-gate.c` (card), `tools/col-ref.c` (host, double). Build: `./build.sh`; both gate binaries audit 0 illegal / 0 suspect.

Commands: `phi run /tmp/cortex/col-gate OUTDIR STEPS THREADS [DUMP_THREAD...]` (env `CX_RINGN` bus matrices cycled, `CX_STRIDE=2` one thread per core, `CX_PMU0/1` raw events); `build/host/col-ref DIR STEPS THREADS DUMP_THREAD...`; `build/host/stats 1238.093 FILE...`.

Raw logs: `results/s01-col-1thr.log`, `-120thr.log`, `-nopf.log`, `-iso.log` (hidden 96), `-h64.log` (hidden 64), `-pmu.log`. Data: `data/col/` (hidden 96) and `data/col/h64/`.

## Kernel

Per step: 5 GEMVs (P_in 128x64, Wqb 64x16, W 80x208, W1 144xH, Wu Hx128, Wb Hx16) as broadcast + FMA with the weight load folded in, `vprefetch0` of the row two ahead; softmax over the 120 bus slots (masked `vexp223ps`, horizontal max/sum by `vmovaps` swizzles and `vpermf32x4` 0xB1 / 0x4E); RMSNorm and k-normalisation by `vrsqrt23ps`; sigmoid = `vrcp23ps`(1 + exp2(-x log2 e)); the fast weights kept as T = S^T so that S k and S q are one broadcast-FMA sweep, with o = a (S q) + (k . q) d, then the rank-1 update sweep. Disassembly of `run()`: 0 vector spills, 82 `vfmadd231ps` sites, 14 `vpermf32x4`, 11 `vgmaxps`, 17 `vexp223ps`.

Trap hit: a `malloc` buffer passed to `_mm512_store_ps` raises a general protection fault on the card (dmesg `traps: col-gate ... general protection fault`); 64-byte alignment is required.

## Wiring check (card F32 vs host double, same weights and inputs)

| Run | Threads checked | max err relative to the vector's max |ref|: u / bc / g / final T |
|---|---|---|
| hidden 96, 1 thread, 10,000 steps | 0 | 1.02e-6 / 1.04e-6 / 1.6e-7 / 3.4e-7 |
| hidden 96, 120 threads, 4,000 steps | 0, 59, 119 | <= 9.2e-7 / 1.09e-6 / 1.6e-7 / 3.4e-7 |
| hidden 64, 1 thread, 10,000 steps | 0 | 8.9e-7 / 1.06e-6 / 1.7e-7 / 3.4e-7 |
| hidden 64, 120 threads, 4,000 steps | 0, 59, 119 | <= 7.7e-7 / 1.12e-6 / 1.7e-7 / 3.4e-7 |

PASS: a wiring fault would show as O(1). The residual is F32 accumulation plus the 1-ULP transcendental instructions. Reference ranges over the synthetic inputs: |u| <= 0.74, g in 0.37..0.65, a and b <= 0.93, bus attention max 0.011 (near uniform, as random inputs give).

## Timing (lockstep = one core2r barrier per step, thread 0 times barrier exit to barrier exit)

| Shape | Column bytes | Threads | Step median us | p99 | mean | L2 read misses / step / thread | Column-steps/s |
|---|---|---|---|---|---|---|---|
| hidden 96 | 235,968 | 1 | 27.5 (own; min 25.3) | 69 | 29.8 | 152 | 30.1k |
| hidden 96, no prefetch | 235,968 | 1 | 81.9 (own) | 147 | 88.5 | 181 | 10.8k |
| hidden 96 | 235,968 | 60 (one per core) | 36.3 | 125 | 42.2 | 153 | 1.42 M |
| hidden 96, ring 1 | 235,968 | 60 | 31.9 | 114 | 37.5 | 32 | 1.60 M |
| hidden 96 | 235,968 | **120** | **150.3, 157.6, 162.6, 181.1** (4 runs) | 274-329 | 164-195 | 978-1,062 | 0.61-0.73 M |
| hidden 96, ring 1 | 235,968 | 120 | 150.1 | 288 | 166 | 879 | 0.72 M |
| hidden 96, no prefetch | 235,968 | 120 | 149.4 | 332 | 170 | 991 | 0.70 M |
| **hidden 64** | 198,592 | 1 | 24.9 (own; min 22.7) | 60 | 26.0 | 143 | 34.5k |
| hidden 64 | 198,592 | 60 (one per core) | 34.5 | 79 | 38.4 | 150 | 1.56 M |
| **hidden 64** | 198,592 | **120** | **79.8, 79.9** (2 runs) | 199-201 | 92.3-92.4 | 303-304 (thread 0: 377-393) | 1.30 M |

Instructions per step (raw 0x0016 INSTRUCTIONS_EXECUTED, hidden 96, 1 thread): 13,409. The VPU-only umask (raw 0x2016) reads 0 under the card kernel's knc.c driver.

## Findings

1. **Two 236 KB columns per core do not stay in the 512 KiB L2.** At 120 threads the misses rose 7x (152 -> ~1,050 per step) and the step 5.5x against one column per core, and keeping the bus matrix constant (ring 1) changed little (879 misses). Two contiguous 236 KB blocks average 7.4 of the L2's 8 ways per set; with the bus, embedding rows, stack and code the sets overflow.
2. **The plan's named first shrink fixes most of it:** MLP hidden 96 -> 64 (column 198,592 B, 2 per core = 78 % of L2): 120-thread step 79.9 us median, misses 303. Column params 44,512 (was 52,657); cluster total 120 x 44,512 + 524,288 + 128 = 5,865,856 [arithmetic].
3. **Two threads per core still cost aggregate rate:** 1.30 M column-steps/s at 120 threads vs 1.56 M at 60 (one per core), i.e. the second thread per core does not pay for itself here (it did in the Stage 0f GEMV, 218 vs 136 GFLOP/s). The remaining 303 misses are the two columns still evicting each other in part. The 120-column layout stays (the request is two threads per core, each its own net); the gap is recorded for Stage 4 to revisit (options: hidden 48, or Wu streamed from GDDR with far prefetch).
4. **Prefetch two rows ahead is worth 3x on an idle core (27.5 vs 81.9 us) and nothing at 120 threads** (misses dominate). Kept.
5. **Per-token bound, hidden 64 [arithmetic from measured parts]:** 79.9 us step (forward + 1 barrier) + 2 x 6.8 us barriers + 0.7 us mailbox = ~94 us median (~106 us mean), ~9.5k-10.6k tokens/s, before the head, softmax and host sampling.

## Gate verdict

PASS for correctness (1e-6). The timing gate is met at hidden 64 with the number above; the 97 us Stage 0f projection (70.7 + barriers) is replaced by ~94 us at the smaller column.
