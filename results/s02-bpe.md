# Stage 2: tokenizer and data (2026-10-25 23:0x, host 5800X)

Code: `tools/bpe.c` (byte-level BPE, GPT-2-style pre-tokeniser on bytes, vocab 4096 = 256 bytes + 3,839 merges + `<|endoftext|>` = 4095). Build: `build.sh` (gcc -O2). Data: `data/tinystories/` (corpus), `data/bpe/` (vocab4096.txt, valid.u16, train.u16).

## Corpus

| File | Bytes | sha256 (local, also re-read with O_DIRECT) | Hugging Face LFS oid |
|---|---|---|---|
| TinyStoriesV2-GPT4-train.txt | 2,227,753,162 | 6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443 | same: MATCH |
| TinyStoriesV2-GPT4-valid.txt | 22,502,601 | 6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d | same: MATCH |

Correction: PLAN.md had recorded c5cf5e22... for the train split; that oid belongs to the V1 file `TinyStories-train.txt` (1,924,281,556 B) in the same listing. Fetched with curl (`data/tinystories/fetch.log`: valid 22:48:45, train 22:50:12).

## Commands and numbers

| Step | Command | Result |
|---|---|---|
| train | `bpe train train.txt 200000000 vocab4096.txt` | 200,000,000 B sample = 48,754,030 words, **29,007 unique**; 3,839 merges in 3.2 s; first merge (" ", "t") count 5,699,370; last merge count 1,038 |
| encode valid | `bpe encode vocab4096.txt valid.txt valid.u16` | 5,681,244 tokens, **3.961 bytes/token**, 148.9 MB/s; ids 10..4095, 27,630 `<\|endoftext\|>` |
| decode valid | `bpe decode vocab4096.txt valid.u16 valid.dec; cmp` | **byte-identical** to the original |
| encode train | `bpe encode vocab4096.txt train.txt train.u16` | **562,693,978 tokens**, 3.959 bytes/token, 174.1 MB/s, 13.7 s; shard 1,125,387,956 B (u16 LE) |

## Findings

1. TinyStories' vocabulary is tiny (29k unique pre-tokens in 200 MB), so the 4096 BPE covers it at ~4 bytes/token; the plan's "~4x fewer steps than bytes" ratio is now measured: 3.96.
2. The whole train split tokenised is 1.13 GB of u16: it fits host RAM (31 GiB) and the plan's 50M-token first slice is the first 100 MB of `train.u16`.
3. Token 4095 (`<|endoftext|>`) separates stories; 27,630 stories in valid, i.e. ~206 tokens per story.
