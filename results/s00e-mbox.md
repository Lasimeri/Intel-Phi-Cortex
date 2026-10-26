# Stage 0e: host <-> card mailbox round trip (2026-10-25 09:08)

Commands:
- card: `setsid nohup /tmp/cortex/mbox-gate 1344 16`. Dispatcher pinned to CPU 241. It polls the host's command word through the uncached `/dev/phihost` map with `delay 16` between reads, writes the payload with 512-bit stores into host memory, then the ACK.
- host: `build/host/mbox-ping data/mbox 10000 5`, pinned to host CPU 2, spinning on ACK. 5 repeats, payloads interleaved.
- `build/host/stats 1000 mbox-*.u32`.

Raw: `results/s00e-mbox.log`, `data/mbox/`.

| Payload | Median ns (5 repeats) | p99 ns | p99.9 ns | max ns |
|---|---|---|---|---|
| 0 (command + ACK) | 700-710 | 1,330-1,530 | 3,300-8,530 | 9,440-35,930 |
| 16 KiB card -> host, 256 uncached 64-B vector stores | 25,540 (all 5) | 33,650-33,940 | 35,170-35,630 | 36,130-72,780 |

- Payload check: first and last word of each 16 KiB payload after the ACK. **0 mismatches** in 50,000 checks: the posted writes arrive in order, payload before ACK.
- Card side: served 100,000 commands in 2.339 s; the session was taken at seq 0, with no replay.

## Findings

1. **The mailbox is cheap.** One command + ACK is 0.70 us median. That is below the ~1 us card read of host memory in Intel's guide, and small next to a 6.8 us barrier.
2. **Uncached card stores to host memory are slow:** (25,540 - 700) ns / 256 lines = ~97 ns per 64-byte store = **0.66 GB/s**. Sending the full 4096 x F32 probability vector this way costs ~25 us per token, more than 3 barriers.
3. **Decision for Stage 5:**
   - The card applies the min-p mask, then compacts the survivors (values + token ids) with KNC `vpackstore` (masked pack store), so only candidates cross PCIe.
   - Bulk card->host blocks go by DMA on channel 2. (This card measured ~3.6 GB/s card->host earlier for 512 KiB copies; small-copy latency is not measured yet.)
