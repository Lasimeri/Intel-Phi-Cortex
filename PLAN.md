# Intel Phi Cortex: plan

Status 2026-10-26 01:0x: **Stages 0 to 5 done** (120 columns train on the card at 1.22k tok/s and generate through the host-driven mailbox + DMA loop at 2.16k tok/s greedy / 1.4-1.5k sampled; after 1M tokens the cluster is not better than one column and the bus hurts: see "Stage 4 results"). Stages 6 (online learning) and 7 (CLI chat / HTTP) not started; `host/cortex` is the CLI for single prompts.
Every number below is tagged: **[measured]** (with its source), **[source]** (document), **[arithmetic]** (derived, to be replaced by a measurement at the named gate), **[unverified]**.

## Request (user, 2026-10-25 ~08:39)

> Start a plan to run a cluster of F32 neural nets written in C on me Xeon Phi 7120A (put the project in its own folder, make sure for a note to use the cards vector processing unit for as much as possible, and all other scalar ops performed on the 5800X, interleaved with the Xeon Phi KNC card)... | I want this setup to function as if it were a standard LLM, however utilizing 120 threads each with a neural net with heuristic capacty to train and generate tokens in real-time, moreso similar to how a brain functions

> Put the project into its own folder on my 16TB HDD

## DIRECTION NOTE (binding for every stage)

1. **The 7120A's 512-bit VPU does as much of the work as possible.** All F32 math (matmul, fast-weight updates, norms, activations, softmax, loss, backward, optimizer) runs on the card as KNC vector instructions (S3 clang `-march=knc`, KNC intrinsics/builtins).
2. **Scalar work runs on the host Ryzen 7 5800X**, never on card cores when avoidable: tokenizer, data streaming from the HDD, sorting/selection for sampling, the final random draw, I/O, scheduling, checkpoints, logs.
3. **Interleaved**: the host works on step t+1 (or t-1) while the card computes step t. One mailbox exchange per token step serves all 120 columns. (User, 2026-10-25 22:5x: "Interleave the instructions that can be run on the card with host CPU only instructions via DMA".) Bulk data between the two sides moves by DMA on the card's channel 2 (token batches and weights in, compacted probabilities out), split at 2 MiB pages; the mailbox words carry only sequence and acknowledgement. Host-only instructions (BPE, top-k/top-p and the draw, I/O, checkpoints) run while the card computes the current step.
4. Card scalar code is limited to loop control, addresses, barriers, mailbox flags and DMA descriptors.
5. F32 throughout (card and host reference).
6. C only (card: S3 clang; host: gcc). No Python anywhere, including tooling and data fetch (curl, not `hf`).
7. No count tables, n-gram lookups or transition statistics as a generator. Every prediction comes from learned weights. (BPE merges are the tokenizer; unigram entropy appears only as a metric.)
8. Each stage ends in something that runs on the 7120A and prints a measured number. Nothing is reported as working before that.

## Interpretation (decided; stated so it can be overridden)

| Phrase | Decision | Why |
|---|---|---|
| "cluster of neural nets, 120 threads each with a neural net" | 120 **columns**: one small recurrent net per KNC hardware thread, each owning its weights and state | Literal mapping; KNC needs >= 2 threads per core to fill the in-order pipe, 60 cores x 2 = 120 |
| "function as if it were a standard LLM" | Text in, BPE tokens out, streamed, through a CLI and an OpenAI-style HTTP endpoint; quality target = TinyStories-class coherent English, not general-assistant quality | Eldan & Li 2023: fluent, consistent stories below 10M parameters, even with one block **[source]**; this cluster totals ~6.84M **[arithmetic]** |
| "how a brain functions" | Thousand Brains layout: every column is a complete predictor; columns reach consensus by voting; lateral long-range links between columns (one step delayed) | Hawkins, Ahmad, Cui 2017 (Front. Neural Circuits 11:81) and Hawkins et al. 2019 (Front. Neural Circuits 12:121): columns learn complete models and vote over long-range connections **[source]** |
| "heuristic capacity to train ... in real-time" | Each column carries fast weights updated every token by the **gated delta rule** with write strength beta_t and forgetting alpha_t computed by the column itself from its input (the column sets its own learning rate and forgetting); slow weights learn online from external text | Yang, Kautz, Hatamizadeh, ICLR 2025 (arXiv:2412.06464): the gated delta rule is test-time SGD on 0.5 ||S k - v||^2 with adaptive rate beta_t and weight decay alpha_t **[source]** |
| "generate tokens in real-time" | Fast weights update during generation; slow weights update only from text that came from outside (prompt, documents fed in). Never from the model's own samples | Learning from its own samples carries no new information and drifts |

## Hardware facts used

| Fact | Value | Tag / source |
|---|---|---|
| Card | Xeon Phi SC7120A, 8086:225c sub 7d99, desktop 0000:2f:00.0 | [measured] lspci 2026-10-21 |
| Cores / threads | 61 x 4 = 244 (nproc 244) | [measured] 2026-10-21 |
| Clock | 1238 MHz | [measured] phitop 2026-10-21 |
| GDDR5 | MemTotal 15,789,924 kB (15.06 GiB) | [measured] 2026-10-21 |
| Link | PCIe 5.0 GT/s x8 (card max x16) | [measured] sysfs 2026-10-21 |
| CPU map | CPU 0 = core 60 thread 0 (BSP); CPU k (1..240) = core (k-1)/4, thread (k-1)%4; CPUs 241-243 = core 60 threads 1-3 | [measured] 2026-10-21, all 244 checked |
| Block pollers | phiblk0-poll CPU 239, phiblk1-poll CPU 238 (core 59) | [measured] 2026-10-21 |
| Core issue | in-order, 2 pipes, no back-to-back issue from one thread (1 thread = at most half the slots) | [source] Intel SDG 328207 |
| Vector latency | 4 cycles op, 7 load/store, 6 convert/shuffle | [source] same |
| L1 / L2 | 32 KiB 8-way / 512 KiB per core, 8-way, inclusive | [source] same |
| GDDR bandwidth | STREAM 165-180 GB/s at 60 threads; 3.6-4.7 GB/s per core alone | [source] same |
| Host <-> card | card read of host memory ~1 us non-posted round trip; card writes to host posted | [source] same |
| F32 transcendentals | `vexp223ps`: input int32 Q8.24 fixed point, 0.99 ULP, needs `vcvtfxpntps2dq` with exponent adjustment first; `vrcp23ps` 0.912 ULP; `vrsqrt23ps` 0.775 ULP; `vlog2ps` 4 ULP (abs < 2^-21 on [0.5, 2]) | [source] Intel 327364-001 pp. 190, 577, 588, 373 |
| Toolchain | local KNC clang has builtins for exp223/log2/rcp23/rsqrt23/cvtfxpnt; no gather/scatter builtins | [measured] grep of BuiltinsX86KNC.td, 2026-10-25 |
| F32 peak | 60 cores x 16 lanes x 2 (FMA) x 1.238 GHz = 2377.0 GFLOP/s; 39.6 per core; 19.8 per thread at 2 threads/core | [arithmetic] |
| Earlier F32 GEMM on this card | 347-437 GFLOP/s on large shapes at 118 threads | [measured] 2026-10-24 (notes; raw log no longer on disk) |
| Host | Ryzen 7 5800X, 8C/16T, AVX2 (no AVX-512), 31 GiB RAM | [measured] 2026-10-24 |
| 16TB HDD | /dev/sda2 ntfs3 on /mnt, USB, 1,178,139,820,032 B free | [measured] df 2026-10-25 08:48 |

Consequence of the ISA facts: in F32 the card does softmax, sigmoid/SiLU, RMSNorm, division and square root itself (one instruction each, plus the convert for exp). No Newton steps are needed at F32.

## Architecture

### Column (one per hardware thread), starting configuration

Per token t, column c:

```
p   = P_in e_t                       e_t: shared embedding row (4096 x 128), P_in 128x64
r   = attn(Wq_bus p, bus_{t-1})      bus: 120 slots x 16, written by all columns at t-1
x   = RMSNorm([p ; r])               80 dims
q,k,v,a,b = W x                      W 80x194; a = sigmoid (forget), b = sigmoid (write strength)
S_t = S_{t-1} a (I - b k k^T) + b v k^T     fast weights 64x64, unit-norm k
o   = S_t q
z   = SiLU(W1 [o ; x])               W1 144x96
u_c = Wu z (vote, 128)   g_c = sigmoid(wg z) (confidence)   b_c = Wb z (bus out, 16)
```

| Item | Size | Bytes F32 |
|---|---|---|
| P_in 128x64 | 8,192 | |
| bus query 64x16 | 1,024 | |
| W 80x194 | 15,520 | |
| W1 144x96 + bias | 13,920 | |
| Wu 96x128 | 12,288 | |
| wg 96 + bias | 97 | |
| Wb 96x16 | 1,536 | |
| RMSNorm gain | 80 | |
| **slow weights** | **52,657** | **210,628 (205.7 KiB)** |
| fast weights S 64x64 | 4,096 | 16 KiB |
| head slice (35 rows x 128) | 4,480 | 17.5 KiB |
| bus read (120 x 16) | 1,920 | 7.5 KiB |

Working set per thread 246.7 KiB for the table rows + ~2 KiB activations = ~248.7 KiB; two threads per core ~497 KiB of 512 KiB L2 **[arithmetic]**. Tight on purpose; the Stage 4 gate decides L2-resident vs GDDR-streamed. First shrink if needed: W1 hidden 96 -> 64 (-9.2k params, -36 KiB).

Cluster: 120 x 52,657 + embedding 524,288 + consensus gain 128 = **6,843,256 params, 27.4 MB F32** **[arithmetic]**.

### Consensus and output

- h = RMSNorm(sum_c g_c u_c); logits = E h with the head tied to the embedding E (4096 x 128).
- Because the head is linear, E (sum_c g_c u_c) = sum_c g_c (E u_c): the consensus equals a confidence-weighted vote of each column's own logits. Each column is a complete next-token model.
- The head is split by rows over the 120 threads (34-35 rows each). Each thread also computes partial max and partial sum of exp for the softmax.
- Barriers per token step **[measured at Stage 0d: 6.8 us median each, core-aware tree with the release passed down]**:
  1. after the votes;
  2. after h;
  3. after the softmax partials.
  - 3 x 6.8 = ~20 us per token, the same order as the compute. Stage 4 should cut this to 2, e.g. every thread reduces h itself after the vote barrier.
- Lateral bus is one step delayed, so a step needs no intra-step communication between columns except these reductions.
- Training treats the bus input as stop-gradient: each column's backward stays on its own thread. This is local learning, like cortex. The Stage 4 gate measures the bus on vs off.

### Modes

| Mode | Input | Fast weights | Slow weights |
|---|---|---|---|
| PRETRAIN | corpus batches (B sequences per column thread) | yes (BPTT, chunked recompute) | Adam on the card |
| READ | external text (prompt, documents), token by token | every token | online step after each token, using the known next token |
| GENERATE | own samples | every token | frozen |

Chat = READ over the prompt, then GENERATE. Context lives in the recurrent and fast-weight state. Total state is 120 x 64 x 64 floats = 1.9 MiB **[arithmetic]**, constant, with no KV cache. There is no hard context limit; recall degrades instead.

## Placement (per op)

| Op | Where | Instructions / note |
|---|---|---|
| embedding row fetch | card VPU | vector loads of row id*128 (scalar address only) |
| all projections (GEMV per token, GEMM in batch) | card VPU | `vfmadd231ps`, broadcast, prefetch via inline asm `vprefetch0/1` |
| gated delta update, S q | card VPU | rank-1 FMA updates, 64x64 in registers/L1 |
| sigmoid, SiLU | card VPU | `vcvtfxpntps2dq` + `vexp223ps`, `vrcp23ps` |
| RMSNorm | card VPU | `vrsqrt23ps` |
| bus attention softmax (120) | card VPU | exp223 sequence |
| consensus sum, head GEMV | card VPU | FMA |
| softmax over vocab | card VPU | max, exp223, sum, `vrcp23ps` |
| temperature | card VPU | `vmulps` |
| repeat penalty | card VPU | host sends a 4096-bit recent-token bitmap (512 B); mask registers from 16-bit chunks, masked multiply |
| min-p filter | card VPU | compare p >= min_p * p_max, masked zero |
| greedy argmax | card VPU | vector max + index |
| loss, backward, Adam | card VPU | `vlog2ps`; Adam via `vrsqrt23ps` + `vrcp23ps` |
| barrier, mailbox flag, loop control | card scalar | minimal; spin with KNC `delay` |
| BPE encode/decode | host | scalar, gcc |
| top-k / top-p selection, random draw | host | scalar sort/partition of the probs the card sent |
| corpus read, shard pack, batch build | host | scalar; double buffered |
| HTTP/CLI, streaming text out | host | |
| checkpoints, logs, gate reports | host | writes to /mnt/Intel Phi Cortex |

Generation defaults: the user's sampling settings (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, repeat penalty 1.2). The greedy run is the correctness baseline.

## Interleave timeline

```
GENERATE / READ, token t:
host: write id_t ─────────────▶ [detokenize + stream t-1] [wait] [GENERATE: top-k/top-p, draw id_t+1 | READ: next known id] write id_t+1
card:            [fwd t: columns ▶ barrier ▶ h ▶ head ▶ softmax ▶ probs to host] [READ: online update t]
                                                          the online update overlaps the host draw
PRETRAIN, batch b:
host: [pack batch b+1 from RAM shards, DMA to card ring] [log loss b-1] [checkpoint every N]
card: [fwd/bwd/Adam on batch b, 120 column threads in parallel]
```

## Card process layout

- One card daemon `cortexd`: 120 column threads plus one dispatcher thread.
- Column threads: cores 0-58 threads 0,1 (CPUs 4c+1, 4c+2); core 59 on CPUs 237 and 240, avoiding the poller CPUs 238/239. Core 60 (CPU 0 and the OS) is left alone except the dispatcher.
- Dispatcher: CPU 241 (core 60 thread 1). It polls the mailbox and releases each step, so the 120 column threads stay identical in timing. Fallbacks if a Stage 0 or 4 gate fails:
  - (a) column 0 doubles as the dispatcher;
  - (b) turn the card swap off, so its poller goes idle (reversible; the user's call);
  - (c) 118 columns.
- Its own region part, its own DMA channel and its own huge-page arena, all picked from what is free at Stage 0 (read the DMA control register and the region layout first).
- GDDR holds weights, Adam m/v, BPTT checkpoints, the double-buffered bus and the token ring. All resident on the card; the shared host region carries only the mailbox.
- Code base: the existing KNC F32 GEMM tile kernel, spin pool and card-DMA code in the user's own repos are **copied** in with a provenance header (not linked). The existing daemon shape (one big GEMM split over the pool) does not fit: here each thread runs single-thread kernels on its own column.
- Known traps carried forward:
  - pool threads must spin with `delay`, not sleep (sleep/wake cost ms per op);
  - a restarted daemon must start with done = posted, or it replays the previous session's lists;
  - DMA must split at every 2 MiB page;
  - `__builtin_prefetch` emits nothing for knc;
  - deploy a running binary as `.new` + `mv` ("Text file busy");
  - phi-isa-audit flags VEX3 `vprefetch0` as CMOV (false positive).

## Data and tokenizer

- Corpus: TinyStoriesV2-GPT4.
  - train 2,227,753,162 B, sha256 6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443 (corrected 2026-10-25 23:0x: the c5cf5e22 value first recorded here is the V1 TinyStories-train.txt oid; verified against the HF LFS listing and the downloaded file);
  - valid 22,502,601 B, sha256 6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d;
  - `Evaluation prompts.yaml` 11,759 B.
  - Sizes and LFS oids from the HF API listing, 2026-10-25 **[source]**.
  - License CDLA-Sharing-1.0 (dataset card) **[source]**.
  - Fetched with curl from `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/<file>` into `data/`. Sizes and sha256 are checked after download.
- Tokenizer: byte-level BPE, vocab 4096 (256 bytes + merges + `<|endoftext|>`), trained by our own C tool on the train split.
  - Chosen over raw bytes: ~4x fewer steps per text **[unverified ratio, measured at Stage 2]**, and the tied head stays 4096 x 128.
  - The encoded corpus goes to `data/` as u16 shards. ~1.1 GB at 4 B/token **[arithmetic]**; it fits host RAM, so the HDD is read once per epoch.
- Metrics:
  - bits per byte (BPB) on the valid split;
  - baselines on the same data and tokenizer: unigram entropy (metric only), 1 column alone, 120 columns with the bus off;
  - the greedy story from fixed prompts, byte-identical across reruns.

## Compute budget [arithmetic, replaced at the gates]

- **Stage 0f measured** a 210 KB column-sized GEMV at 120 threads, with per-column staggered offsets and L1 prefetch.
  - Free-running threads: **51.6 us median per thread** (p99 94-96 us). That is 6.6x the 7.8 us peak arithmetic below.
  - **Lockstep, with a barrier per step as cortexd will run: 70.7 us median step** (p99 134-142, mean 77).
  - Per-token bound: ~85 us median (70.7 + 2 x 6.8 + 0.7) [arithmetic from measured parts; head, softmax and sampling not yet included].
- Forward per token per column ~72.7k MAC, plus 4.5k MAC of head slice, = ~154k FLOP. At 19.8 GFLOP/s per thread peak that is 7.8 us; at 25 % of peak, 31 us.
- READ mode adds ~2x forward for the online step.
- Token-step ceiling ~10^4 tok/s before mailbox and barrier costs. Speed is not the constraint; training quality is.
- Pretraining ~55 MFLOP per token. ~5.6e8 tokens at 4 B/token gives ~3.1e16 FLOP per epoch: 28 h at 300 GFLOP/s sustained, 8.5 h at 1 TFLOP/s.
- First run uses a 50M-token slice (~2.5 h at 300 GFLOP/s).

## Stages and gates

Each gate writes `results/sNN-<name>.md` with the command, the numbers, the spread over repeats and the raw log path.

| Stage | Builds | Gate (printed numbers) |
|---|---|---|
| 0 Facts | folder, build script, mount guard | card up (nproc 244, MemTotal); **card processes already running and the CPU sets and arena memory they hold** (`phi run ps` + each PID's affinity); if cores 0-60 are occupied, report it and stop (freeing them is the user's call, never stop another daemon); free DMA channel and region part; free huge pages; poller CPUs re-read. **ISA sweep**: card exp2 (cvtfxpnt + exp223), rcp23, rsqrt23, log2 vs host libm double over 2^20 points, max ULP per op. 120-thread spin barrier median/p99 over 10^5 rounds. Mailbox round trip median/p99 over 10^4. Per-thread loop rate with 248 KiB resident in L2 vs streamed from GDDR. Card binary builds and runs from /mnt (ntfs3 exec). |
| 1 One column, forward | `col.c` VPU kernels: GEMV, gated delta, RMSNorm, SiLU, sigmoid, bus attention | one wiring check vs a host double reference (max rel err); `phi-isa-audit` clean; VPU instruction count (perf 0x2016); us per step on one thread |
| 2 Tokenizer + data | `bpe.c` train/encode/decode, shard writer, curl fetch + sha256 | sha256 match; decode(encode(valid)) byte-identical; vocab 4096; bytes/token; encode MB/s; shard bytes |
| 3 One column learns | BPTT with chunked recompute; Adam on VPU | finite-difference gradient check (host, one batch); loss curve on 10M tokens; valid BPB vs unigram BPB |
| 4 120 columns, pretrain | cortexd threads + dispatcher, consensus, bus, batched training, DMA token ring | tokens/s, sustained GFLOP/s; valid BPB every N steps vs the stage 3 single column; bus on vs off; L2-resident vs GDDR step time; core 59 columns vs the rest; g_c usage spread (share of total confidence per column) |
| 5 Generation | mailbox token loop, card softmax/min-p/penalty, host top-k/top-p/draw | greedy text for the fixed prompts = recorded baseline (byte-identical on rerun) and its tok/s, repeated and interleaved; then sampled text with the default settings |
| 6 Real-time learning | READ online step, GENERATE fast-only | adaptation: BPB on the second half of unseen valid stories after reading the first half (online on vs off); forgetting: valid BPB before/after 1M online tokens; tok/s with learning on vs off, interleaved repeats. The slow-weight online rule (plain SGD vs surprise-momentum with forget gate, after Titans arXiv:2501.00663) is chosen from these numbers |
| 7 LLM interface | `cortex` CLI chat; HTTP on 127.0.0.1 with OpenAI-style `/v1/completions` and `/v1/chat/completions` (SSE streaming); checkpoint save/load | an existing OpenAI-style client streams a completion; checkpoint round trip reproduces the stage 5 greedy text byte for byte |

## Stage 0 results (2026-10-25 09:00-09:17, all on the 7120A)

| Gate | Result | Report |
|---|---|---|
| 0a facts | 244 CPUs; MemFree 11.13 GB; 2052/2052 huge pages free; pollers CPU 238/239 mostly asleep (12 ticks per 5 s); region part 1344 MiB free | `results/s00a-facts.md` |
| 0b toolchain | S3 clang + knc-cc build, audit clean, runs on the card; TSC = 1238.093 MHz (core clock); DCR 0x169d, channel 2 free; ntfs3 exec works | `results/s00b-hello.md` |
| 0c ISA | all within Intel's bounds over 2^20 inputs: exp2 instruction 0.9366 ULP (end-to-end 1.165), rcp23 0.8991, rsqrt23 0.704, log2 1.038 ULP / abs 9.5e-08 | `results/s00c-isa.md` |
| 0d barrier | 120 threads, core-aware tree with the release passed down: 6.76-6.97 us median, p99 7.5-8.1 us; one shared release line is unstable (10.1 then 13.2-16.0 us); central counter 17.8 us | `results/s00d-barrier.md` |
| 0e mailbox | command + ACK 0.70 us median (p99 1.3-1.5 us); 16 KiB by uncached card stores 25.5 us (0.66 GB/s): probabilities must be compacted on the card (min-p + vpackstore) and/or sent by DMA | `results/s00e-mbox.md` |
| 0f L2 vs GDDR | identically aligned per-thread blocks evict each other from L2 (50-58 % misses at 120 threads). With per-column staggered offsets: 51.6 us per 210 KB GEMV free-running, 218.5 GFLOP/s aggregate; **lockstep step 70.7 us median** (118 threads: 69.6-69.9, so keep 120); L2-resident 2.4x faster than GDDR-streamed (126 us) | `results/s00f-l2.md` |

Design changes from Stage 0:
1. Columns stay L2-resident, and each column's weights get a distinct offset mod 2 MiB (prime line stride). Hot loops prefetch into L1 two rows ahead.
2. Barriers use the core-aware tree with the release passed down (`card/team.h`, CX_CORE_REL, arity 2). Cut from 3 to 2 per token at Stage 4.
3. Card->host probabilities: compact on the card first. Never ship 16 KiB by uncached stores per token.
4. F32 softmax/sigmoid/SiLU/RMSNorm/Adam stay on the VPU (no Newton steps).

## Stage 1 results (2026-10-25 22:5x-23:2x, all on the 7120A; `results/s01-col.md`)

| Gate | Result |
|---|---|
| wiring (card F32 vs host double, same weights and inputs, 10,000 steps) | max error <= 1.1e-6 relative to each vector's largest value, for u, bc, g and the final fast weights; at 120 threads threads 0, 59 and 119 checked |
| isa-audit | 0 illegal, 0 suspect |
| instructions per step (raw 0x0016; the VPU umask 0x2016 reads 0 under knc.c) | 13,409 at hidden 96 |
| us per step, one thread alone | 27.5 median (hidden 96), 24.9 (hidden 64); without the two-rows-ahead prefetch 81.9 |
| us per step, 120 threads in lockstep | **hidden 96: 150-181 median, ~1,050 L2 misses per step (two 236 KB columns do not stay in the 512 KiB L2)**; **hidden 64: 79.9 median, 303 misses**; one column per core: 34.5-36.3 |

Design changes from Stage 1:
1. MLP hidden 96 -> 64 (`CX_HID`, lib/cx-col.h). Column 44,512 slow params = 178,048 B; with T and activations 198,592 B; two per core = 78 % of L2. Cluster total 5,865,856 params [arithmetic].
2. Weight rows are prefetched two ahead (`vprefetch0`): 3x on an idle core, neutral at 120 threads.
3. Two threads per core deliver 1.30 M column-steps/s against 1.56 M for one per core at this column size: the second thread does not pay for itself yet. Kept at 120 columns per the request; Stage 4 revisits (hidden 48, or Wu streamed from GDDR).
4. Per-token bound now ~94 us median [arithmetic from measured parts: 79.9 + 2 x 6.8 + 0.7], before head, softmax and host sampling.
5. The host/card split follows the DMA-interleave direction (DIRECTION NOTE 3): channel 2 rings in both directions, mailbox words for sequencing only.

## Stage 2 results (2026-10-25 23:0x, host; `results/s02-bpe.md`)

| Gate | Result |
|---|---|
| sha256 | train and valid match the HF LFS oids (train's recorded value corrected: it was V1's) |
| decode(encode(valid)) | byte-identical (cmp), 5,681,244 tokens |
| vocab | 4096 = 256 + 3,839 merges + `<\|endoftext\|>`; trained on a 200 MB sample (29,007 unique pre-tokens) in 3.2 s |
| bytes/token | 3.961 (valid), 3.959 (train) |
| encode MB/s | 148.9 (valid), 174.1 (train, 13.7 s for 2.2 GB) |
| shard bytes | train.u16 1,125,387,956 B = 562,693,978 tokens; valid.u16 11,362,488 B |

## Stage 3 results (2026-10-25 23:0x-23:4x; `results/s03-train.md`)

| Gate | Result |
|---|---|
| finite-difference check of the double reference | all 9 groups within 8e-6 relative (E: 8e-10 absolute on a 3.8e-7 gradient) |
| card gradients vs the reference (L 16 and 32) | every group <= 1.1e-6 relative, including all 524,288 embedding entries |
| loss curve | 200k tokens: 12.15 -> 6.43 bits/token (train, last 500 chunks); 2M-token run in progress |
| valid vs unigram | **6.549 bits/token = 1.653 bits/byte** after 200k tokens; unigram 8.433 bits/token = 2.129 bits/byte |
| speed | 773 tokens/s for one column on one thread (1.29 ms per token, head passes 72 % of it) |

Design changes from Stage 3:
1. The embedding lives on the card in vocabulary blocks ETB[v/16][c][v%16] (8 KiB per block); the head and its two gradient passes walk blocks sequentially; a token's row is one block, prefetched one step ahead.
2. Parameter gradients are chunk-end outer products over saved per-step vectors (one write per chunk), not per-step rank-1 updates; the embedding gradient is one block-wise pass plus the input rows.
3. Adam runs on the VPU per chunk (bias corrections folded into lr_t and eps_t; sqrt(v) = s rsqrt(s) with s = v + 1e-38).
4. Once-consumed data is written with plain stores into lines prefetched for writing, never with vmovnraps; the head split across threads is what makes training fast, not single-thread tuning.

## Stage 4 results (2026-10-25 23:5x - 2026-10-26 00:5x; `results/s04-cluster.md`)

| Gate | Result |
|---|---|
| cluster reference vs finite differences (N 4, external bus) | all groups <= 1.0e-5 relative; wg, bg carry gradients; Wb none (stop-gradient) |
| dropped bus-path size (live bus, FD vs analytic) | same order as the local gradient for some entries (Wb true 4.8e-3 vs 0; a P_in entry -1.18e-2 vs -1.40e-2) |
| card cluster forward vs reference (120 columns, live bus) | loss 7.6e-7 abs, hc 3.8e-7, u/bc/g of columns 0, 59, 119 <= 4.0e-7 |
| card cluster backward vs reference | all column groups of columns 0, 59, 119 <= 8.6e-7; cg 5.1e-7; E (524,288) 4.1e-7 |
| timing, idle card, L 32 | forward 355.6-391.9 us/step (collectives + lockstep ~180 of it; T snapshot ~17); backward + gradients 394.5-403.4; training ~775 us/token = ~1.3k tok/s [arithmetic], 4c measures it |
| pretraining, bus on vs off | in progress (1M tokens each, lr 3e-3, valid 100k) |

Design changes from Stage 4:
1. The per-step bus is an array bus[t] (both on the card and in the reference): a ring overwrote what the backward reads (the reference had that bug; the card never did).
2. Collectives per step: vote by 12 group leaders x 10 (2 barriers), softmax by (max, sum) pairs read by every thread (1 barrier); dhc by the same 2-level sum per chunk; input-row embedding gradients summed by the block owner.
3. The row-major shadow ES of the embedding for the input and target lookups (8 lines per step instead of 256), refreshed per chunk by each block owner (built, unverified).
4. Open: the training forward costs 180 us per step at 120 threads against 80 us for the plain forward (the per-step streams evict the two resident columns), and the collectives another 180; measure L2 misses per step (counter added), then cut barriers (fold the pair exchange into the next vote barrier) and streams.

## Stage 5 results (2026-10-26 00:5x-01:0x; `results/s05-gen.md`)

| Gate | Result |
|---|---|
| greedy text for a fixed prompt, byte-identical on rerun | yes (`cmp` of the ids; card-side `cluster-gate gen` and the host-driven `cortex --greedy` agree byte for byte) |
| checkpoint round trip | `cortex` stages the 23.5 MB checkpoint in the host part; the card pulls it by DMA in 12.1-12.2 ms; the greedy text equals the card-side baseline |
| card us per token (mailbox + cluster forward + 16 KiB logits DMA + ack) | 370-450 |
| tok/s overall | greedy 2,159; sampled 1,383-1,489 (host sampler 322-342 us per token, a full sort) |
| sampled text with the user's settings | TinyStories register with grammatical drift (1M tokens of training) |

The host/card split: the card runs the 120 columns, vote, consensus, head and the logits DMA; the host does the sampling chain, the BPE in and out and the printing, and the checkpoint staging. The card's own softmax and min-p are not used in this loop (the host needs the full distribution for top-p anyway); the DMA direction of the user's instruction is honoured in both directions (checkpoint in, logits out).

## Layout of this folder

```
/mnt/Intel Phi Cortex/
  PLAN.md      this file
  card/        cortexd.c, col.c (VPU kernels), vpu.h (exp/sigmoid/rsqrt helpers), cdma.[ch] (copied)
  host/        cortex.c (mailbox client, modes, sampler), bpe.c, shards.c, http.c
  lib/         cx-proto.h (mailbox + op layout shared by host and card)
  tools/       isa-gate.c, barrier-gate.c, col-check.c, gradcheck.c (C, tcc/gcc)
  data/        corpus, sha256 files, u16 shards
  ckpt/        checkpoints
  results/     gate reports + raw logs
```

## Risks and open items

- **Coherence of a 120-column vote: MEASURED at 1M tokens, no gain** (cluster 6.31 bits/token with the bus on, 6.18 off; one column 6.15 train at the same tokens). Candidate causes and next experiments are in results/s04-cluster.md. Original note: The TinyStories evidence is for monolithic transformers. Stage 4 compares against a single column. If the cluster cannot beat it clearly, the next step up is GDDR-streamed columns (e.g. 256k params each = ~31M total).
- **Barriers and the mailbox are unmeasured on this card.** The lateral bus couples all columns at every time step, so a 120-thread barrier runs per generated token AND per BPTT step in pretraining. The 28 h/epoch figure leaves that cost out. Barriers dominate if they cost more than the ~10-30 us of compute per step.
- **L2 budget** is 497 of 512 KiB per core **[arithmetic]**; code, stack and the embedding row share it. Stage 0f proved residency only at 411 KiB per core (one 210 KB GEMV per thread, still 78 misses per pass). Stage 1 must re-measure at the real ~497 KiB working set and the real kernel mix (smaller GEMVs plus the 64x64 delta update).
- **Core 59** shares issue slots with the block pollers if those spin.
- **Online slow-weight learning** can forget the pretraining; Stage 6 measures that before the mode is enabled by default.
- **Mount**: /mnt is ntfs3 on a USB HDD, not in fstab, and the volume is flagged dirty. Every script checks `findmnt -no SOURCE /mnt` = `/dev/sda2` before writing. Otherwise the writes land on the near-full root fs.
- **git**: not initialised; no commit or push unless asked.

## Sources

- Intel Xeon Phi Coprocessor ISA Reference Manual 327364-001 (2012-09-07), https://www.intel.com/content/dam/develop/external/us/en/documents/327364001en.pdf
- Intel Xeon Phi Coprocessor System Software Developers Guide 328207-002
- Eldan, Li, "TinyStories", arXiv:2305.07759
- Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025, arXiv:2412.06464
- Hawkins, Ahmad, Cui, "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World", Front. Neural Circuits 2017, doi:10.3389/fncir.2017.00081
- Hawkins, Lewis, Klukas, Purdy, Ahmad, "A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex", Front. Neural Circuits 2019, doi:10.3389/fncir.2018.00121
- Behrouz et al., "Titans", arXiv:2501.00663
- TinyStories dataset card and file listing, https://huggingface.co/datasets/roneneldan/TinyStories (2026-10-25)

## Design diagram

```
╔═ Desktop host: Ryzen 7 5800X (8C/16T, AVX2, no AVX-512), 31 GiB RAM ═══════════════════════════╗
║ ┌ cortex host (C, gcc) ─────────────────────────┐   ┌ 16TB HDD /mnt (ntfs3, USB) ────────────┐ ║
║ │ BPE 4096 tokenizer / detokenizer              │   │ /mnt/Intel Phi Cortex                  │ ║
║ │ sampler: top-k, top-p, rep-penalty, draw      │   │ data/ TinyStoriesV2 train 2.23 GB      │ ║
║ │ data loader: u16 shards, double buffer        │◀──│ data/ u16 BPE shards (size: Stage 2)   │ ║
║ │ CLI + HTTP 127.0.0.1 (OpenAI-style API)       │──▶│ ckpt/ weights, results/ gate logs      │ ║
║ │ mode control: PRETRAIN / READ / GENERATE      │   │                                        │ ║
║ └───────────────────────┬───────────────────────┘   └────────────────────────────────────────┘ ║
╚═════════════════════════╪══════════════════════════════════════════════════════════════════════╝
                          │ mailbox, one exchange per token step for all 120 columns:             
                          │ host -> card: token id, mode, batch slot   card -> host: probs 16 KiB 
                          │ PCIe Gen2 x8 (5.0 GT/s x8, sysfs 2026-10-21)                          
╔═ Xeon Phi 7120A (KNC) ══╪═ 61 cores x 4 thr, 1.238 GHz, L2 512 KiB/core, GDDR5 15.06 GiB ══════╗
║ ┌───────────────────────▼ dispatcher: CPU 241 (core 60, thread 1) ───────────────────────────┐ ║
║ │ polls the mailbox, releases one step for all 120 columns, spins with KNC delay (no sleep)  │ ║
║ └───────────────────────┬────────────────────────────────────────────────────────────────────┘ ║
║ ┌ 120 column threads ───▼────────────────────────────────────────────────────────────────────┐ ║
║ │ cores 0-58: CPUs 4c+1, 4c+2   core 59: CPUs 237, 240   (238/239 = block pollers)           │ ║
║ │ column c (1 thread, ~205 KiB F32 weights, L2-resident target):                             │ ║
║ │   e_t (shared emb 4096x128) -> P_in 128x64 -> bus read (120 x 16, from step t-1)           │ ║
║ │   -> q,k,v,a,b -> S_t = S_t-1 a (I - b k k^T) + b v k^T   (fast weights 64x64)             │ ║
║ │   -> o = S_t q -> MLP 144->96 SiLU -> vote u_c (128), conf g_c, bus b_c (16)               │ ║
║ └───────────────────────┬────────────────────────────────────────────────────────────────────┘ ║
║ ┌ consensus + head ─────▼────────────────────────────────────────────────────────────────────┐ ║
║ │ h = RMSNorm(sum_c g_c u_c)   logits = E h (tied head, 35 rows per thread)                  │ ║
║ │ softmax: vcvtfxpntps2dq + vexp223ps, sum via vrcp23ps; min-p mask -> probs to host         │ ║
║ └───────────────────────┬────────────────────────────────────────────────────────────────────┘ ║
║ ┌ GDDR5 arena ──────────▼────────────────────────────────────────────────────────────────────┐ ║
║ │ weights, Adam m/v, BPTT checkpoints, bus x2, token ring (own region part + DMA channel)    │ ║
║ └────────────────────────────────────────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
```
