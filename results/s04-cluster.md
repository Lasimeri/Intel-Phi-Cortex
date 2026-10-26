# Stage 4: 120 columns train together (2026-10-25 23:5x - 2026-10-26 00:xx host time)

Code: `tools/cref.h` (host double reference of the cluster step: N columns, shared E and cg, vote hs = sum g_c u_c, hc = RMSNorm(hs) cg, tied head, live lateral bus = every column's bc from the previous step, stop-gradient; `CREF_BUS_EXT` for the finite-difference check), `tools/ctrain-ref.c` (FD), `card/cluster.h` (the card cluster: per-thread column from card/train.h, shared ETB/GET/Adam state, bus[(L+1)][16][128], 2-level vote reduction, (max, sum) pair softmax exchange, 2-level dhc reduction, owner-summed input rows, local Adam), `tools/cluster-gate.c` (fwd / bwd checks, `run` trainer with `CX_BUS=0|1`), `tools/ctrain-check.c` (host comparison). Logs: `results/s04a-ctrain-ref.log`, `s04a-fwd.log`, `s04b-bwd.log`, `s04b-bwd-partial.log`, `s04-timing.log`. Data: `data/s4/`.

## Layout on the card

- 120 threads = 2 per core on cores 0-59 (CPUs 4c+1, 4c+2; core 59 on 237 and 240). Thread k owns column k and the vocabulary blocks k, k+120 and, for k < 16, 240+k (256 blocks of 16 tokens).
- Per thread 4 MiB of huge pages: the hot column (weights 174 KB, T 16 KB, act 6.7 KB) at a prime-stride offset in the first 2 MiB (Stage 0f rule); the streams (saves, gradients, Adam m/v) in the second.
- Shared 8 MiB + 5 x 2 MiB: ETB, GET, m_et, v_et, the row-major shadow ES, the per-step bus, votes, group sums, pairs, dhc partials, de partials.
- Per step 3 barriers: A after the bus write and vote, B after the 12 group leaders summed 10 votes each, C after every thread's local head (own blocks: logits, local max, exps, local sum). Every thread then reads 12 group vectors (hs, identical bits everywhere) and 120 (max, sum) pairs (M, S, loss).
- Per chunk 4 more barriers: D/E (dhc partials, 2-level), F (de partials published, block owners add the rows), G (Adam done; thread 0 updated cg).

## 4a: the host reference against finite differences (`ctrain-ref 4 6 6 12`, external bus)

All groups of columns 0 and 3 within 1.0e-5 relative (E 4.7e-5 on a 1.3e-5 gradient); wg and bg now carry gradients (the vote path), Wb none (the bus is stop-gradient).

With the live bus, FD measures the true gradient including the path through the bus, which the design drops (local learning): column 0, Wb true gradient 4.8e-3 vs analytic 0; a P_in entry -1.18e-2 (true) vs -1.40e-2 (analytic); a W entry 8.1e-4 vs 5.7e-4. The dropped path is of the same order as the local gradient for some entries. Decision kept; bus on vs off is measured in 4c.

A reference bug found by the card: cref.h first kept the live bus in a 2-entry ring and saved per-step pointers into it, so the backward of steps older than the last two read overwritten buses (Wqb "errors" of 0.39-0.58 and P_in 2.3e-3 that were the reference's). Fixed to a per-step bus array like the card.

## 4a/4b: the card against the reference (120 columns, from zero, warm-up chunk of 32 + a chunk of 32, live bus)

| Quantity | max rel err (card F32 vs host double) |
|---|---|
| per-step loss | 7.6e-7 absolute (mean 8.594224 both) |
| hc (32 x 128) | 3.8e-7 |
| u, bc, g of columns 0, 59, 119 | <= 4.0e-7 |
| P_in, Wqb, gain, W, W1, b1, Wu, wg, bg of columns 0, 59, 119 | <= 8.6e-7 |
| cg | 5.1e-7 |
| E (all 524,288 entries: head part and input rows) | 4.1e-7 |

## Timing (idle card, `cluster-gate fwd|bwd valid.u16 s4t 32 32`, L = 32)

| Phase | us per step | Runs |
|---|---|---|
| cluster forward (column fwd with saves, bus, vote, consensus, own-block head, softmax exchange; 3 barriers) | 355.6, 357.7, 359.9, 364.6, 369.1, 391.9 | 6 |
| ... without the T snapshot (`-DCX_NOSNAP`, timing only) | 345.2, 350.8 | 2 |
| ... column forward with saves only, free-running (`-DCX_NOCOLL`) | 179.8, 180.0 | 2 |
| backward + all gradients, no Adam | 394.5, 403.4 | 2 |
| (contaminated, run 2 on thread 0's CPU) forward / backward | 743-747 / 405 | 3 |

So the collectives and the lockstep they impose cost ~180 us, the T snapshot ~17 us, and the training forward with its per-step streams costs 180 us against 80 us for the plain inference forward of Stage 1 at the same 120 threads. Training ~775 us per token [arithmetic from the two measured halves plus Adam] = ~1.3k tokens/s for the whole cluster; to be replaced by the 4c measurement.

## 4c: pretraining (in progress)

`CX_BUS=1 cluster-gate run train10m.u16 run4on 32 1000000 3e-3 valid.u16 100000 250`, then `CX_BUS=0` (run4off). Results appended when done.

### 4c measured (2026-10-26 00:3x-00:5x host; `results/s04c-run4.log`; checkpoints `data/s4/run4on/ckpt.bin`, `run4off/ckpt.bin`, 23,463,424 B each)

`CX_BUS=1|0 cluster-gate run train10m.u16 run4on|off 32 1000000 3e-3 valid.u16 100000 250` (Adam b1 0.9, b2 0.95, lr 3e-3, 50-chunk warm-up, chunks of 32, 120 columns, 5,865,856 parameters).

| | bus on | bus off |
|---|---|---|
| tokens/s (steady, whole cluster, training) | **1,224-1,230** | **1,236-1,239** |
| us per token: forward + vote + head / backward + gradients + Adam | 375-377 / 437-440 | 367-368 / 440-441 |
| train loss, bits/token (250-chunk means) at 64k / 256k / 512k / 1M tokens | 6.92 / 6.54 / 6.41 / 6.38 | 6.56 / 6.09 / 6.12 / 6.13 |
| **valid, 100,000 tokens, after 1M** | **4.3708 nats = 6.3057 bits/token = 1.5919 bits/byte** | **4.2854 nats = 6.1825 bits/token = 1.5609 bits/byte** |
| wall | 816.1 s train + 37.0 s valid | 808.0 s + 36.2 s |

Reference points on the same tokenizer: unigram 8.433 bits/token; one column (Stage 3 run 2, 791 tok/s) 6.23 bits/token on 200k valid tokens after 2M tokens, train loss ~6.3-6.4 at the 1M-token mark.

Findings:
1. **The cluster learns on the card at ~1.2k tokens/s** (120 columns, 5.9M parameters, every column updated every chunk).
2. **The lateral bus hurts at 1M tokens:** off is 0.123 bits/token better than on. The bus carries each column's bc = Wb z with Wb never trained (stop-gradient), i.e. a random projection of the other columns' hidden states; the attention over it adds noise to x that the columns have to learn to ignore.
3. **The 120-column vote is not clearly better than one column at equal tokens:** cluster-on 6.31 vs the single column's ~6.3 train loss at 1M (its valid 6.23 came after 2M tokens). The plan's open risk ("coherence of a 120-column vote is untested") is now measured as "no gain yet". Candidate causes: all columns see the same token stream from different random inits and are trained by one shared loss through one consensus norm, so nothing forces specialisation; the untrained bus; 1M tokens is short.
4. Single-column train loss at the 1M-token mark (run 2): 6.158 / 6.151 bits/token (chunks 30,000 / 32,000) against the cluster's 6.38 (bus on) and 6.13 (bus off): one column with 0.57M parameters matches 120 columns with 5.9M at equal tokens.

### Shadow lookup and miss counts (`results/s04-es.log`, 2026-10-26 00:5x)

The row-major shadow ES (8 lines per lookup instead of 128) leaves every gradient unchanged (<= 5.1e-7) and the forward at 373.0 us: no gain. The counter (raw 0x10cb L2_READ_MISS, first perf event per thread) reads **2,485-2,499 misses per step per thread (mean), 3,064-3,128 on thread 0**, i.e. ~160 KB re-fetched per thread per step, about half of a column's 199 KB; instructions per step 41,320 (mean), 46,699 (thread 0) against ~12,000 for the plain forward. The per-step streams (act copy-out 104 lines, T snapshot 256, bus 128, own ETB blocks 384, pairs 120, group sums 96, prefetch-for-write 360) churn the 512 KiB L2 that also has to hold two 199 KB columns, so the columns are evicted every step. The remedies are structural: fewer resident bytes per core, fewer streamed lines (T checkpoint every 8 steps, 2-level pairs), fewer barriers (fold the softmax exchange into the next vote barrier), or one column per core for training. One unexplained reading: the forward inside the `bwd` gate run measured 689.6 us against 373.0 in the `fwd` run of the same binary a moment earlier.
