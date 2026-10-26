# Stage 5 (part 1): greedy generation on the card (2026-10-26 00:5x host)

`cluster-gate gen CKPT PROMPT.u16 NPROMPT NGEN OUT.u16` (card/cluster.h `cl_argmax`: the thread whose local max equals the global max publishes the token; one extra barrier per token). Logs `results/s05-gen.log`; ids and text in `data/s4/gen-*.u16|txt`. Decoded with `bpe decode`.

| Checkpoint (bus) | Prompt | Greedy continuation (64 tokens) | us/token |
|---|---|---|---|
| run4on (on) | Once upon a time, there was a little girl named | ". She loved to play with her toy. She was very happy. She was very happy. She was very happy. One day, a big, came with her mom. She was very happy. She was very happy. <\|endoftext\|> Once upon a time, there was a little girl named Tim. She" | 367.8, 354.4 |
| run4off (off) | same | " Sue. One day, a little girl named Sue went to the park. She was very happy. She was very happy. She was very happy. One day, a little girl named Sue went to the park. ..." | 377.6 |
| run4on (on) | The dog saw a cat. The cat | " was very happy. The bird was very happy. <\|endoftext\|> Once upon a time, there was a little girl named Tim. She loved to play with her mom. ..." | 407.2 |
| run4off (off) | same | " was very happy. <\|endoftext\|> Once upon a time, there was a little girl named Sue. She loved to play with his toy. One day, she saw a big, red ball. She was very happy. ..." | 347.1 |

- Determinism: the bus-on, prompt-1 run repeated gives byte-identical ids (`cmp`).
- A checkpoint must be generated with the bus setting it was trained with: the bus-off checkpoint fed a live bus produced word salad (gen mode first ignored CX_BUS; fixed).
- Greedy decoding shows the usual repetition loop ("She was very happy.") of a small model; the sampled mode with the user's settings (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, rp 1.2) is the host side of Stage 5.

## Part 2: host-driven loop over the mailbox and DMA (2026-10-26 01:0x host; `results/s05-host-greedy.log`)

`card/cortexd.c` (120 column threads; thread 0 polls the mailbox in the shared host-memory part at 1344 MiB and drives DMA channel 2) and `host/cortex.c` (session, checkpoint staging, sampler in llama.cpp's order, BPE in and out). Protocol in `lib/cx-proto.h`: LOAD (the card pulls 120 weight blocks + cg + the embedding from the staged copy by DMA into each thread's huge page), BUS, RESET, STEP(token) answered with the 4096 raw logits pushed by DMA into the part with the ACK as the DMA status word.

| Measurement | Value |
|---|---|
| checkpoint load by DMA (23,463,424 B, 121 + 3 copies) | 12.2 ms |
| greedy ids via the host (prompt 1, 64 tokens) vs the card-side greedy run | byte-identical (`cmp`) |
| tokens/s overall, greedy | 2,158.9 |
| per token: card (mailbox write, cluster forward, logits DMA, ack) / host (argmax + print) | 449.9 us / 15.4 us |

### Sampled generation through the daemon (`results/s05-host-sampled.log`, settings temp 0.5, top-k 75, top-p 0.95, min-p 0.05, rp 1.2 over the last 64 tokens, llama.cpp order)

| Checkpoint | Seed | tok/s overall | card us/token | host us/token (sort of 4096 + sampling + print) |
|---|---|---|---|---|
| run4on (bus on) | 1 | 1,458.4 | 377.0 | 332.8 |
| run4on | 2 | 1,382.6 | 405.7 | 342.4 |
| run4off (bus off) | 1 | 1,489.2 | 373.3 | 321.5 |
| run4off, prompt 2 | 3 | 1,473.5 | 369.8 | 325.7 |

Sample (run4off, seed 1): "Once upon a time, there was a little girl named Sue. Max loved to play with his toy. One day, her mom saw the bird, "I have a big tree." The she heard it in the room. She saw what he was not fun. They all played together every day. The cat had a new friend and scared, but they could not find their friends." TinyStories register, grammatical drift after a clause or two: the 1M-token, 5.9M-parameter state of the cluster.

The host sampler's full qsort of 4096 candidates costs as much as the card step; a partial selection (top-k by a heap or nth_element) would cut it to ~20 us [estimate].
