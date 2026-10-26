# Stage 0d: 120-thread barrier cost on the 7120A (2026-10-25 09:04-09:07)

Commands:
- `phi run /tmp/cortex/barrier-gate /tmp/cortex/bar 100000 5`: 120 threads pinned by `cx_cpu()`, 1000 warm-up rounds plus 100,000 measured rounds per variant, variants interleaved within each of 5 repeats.
- `build/host/stats 1238.093 bar-*.u32`.

Measure: the TSC ticks on thread 0 between consecutive barrier exits, i.e. one full barrier cycle (all arrivals + release).

Raw: run 1 `results/s00d-barrier.log` + `data/bar/`; run 2 `results/s00d-barrier-2.log` + `data/bar2/`.

## Run 1 (5 shapes)

| Variant | Median us (5 repeats) | p99 us | p99.9 us |
|---|---|---|---|
| central counter | 17.75-17.78 | 38.3-38.7 | 48.9-49.5 |
| tree2, one release line | 10.73 | 28.6-29.1 | 40.5-42.1 |
| tree4, one release line | 10.10 (10,096-10,102 ns) | 25.7-26.9 | 39.0-41.3 |
| tree8, one release line | 10.48-10.49 | 27.1-28.3 | 39.8-40.8 |
| flat flags (thread 0 polls 119) | 12.01 | 29.7-30.9 | 40.4-42.7 |

## Run 2 (adds release-down-tree shapes)

| Variant | Median us (5 repeats) | p99 us | p99.9 us |
|---|---|---|---|
| tree4, one release line | 13.25-16.03 | 34.5-35.3 | 43.9-45.1 |
| tree4r, release down the tree | 8.53-8.62 | 11.4-13.4 | 38.0-38.9 |
| **core2r**, core-aware (partner on the same core, then arity 2 over cores, release down) | **6.76-6.97** | **7.5-8.1** | 36.1-37.3 |
| core4r, the same with arity 4 | 6.91-6.96 | 7.7-9.3 | 35.6-36.4 |
| central (repeat 0) | 17.27 | 37.8 | 48.2 |
| flat (repeat 0) | 15.60 | 35.5 | 44.5 |

## Findings

1. **Larger than PLAN.md assumed.** PLAN.md treated barriers as small next to the ~10-30 us per-token compute. The best barrier costs 6.8 us median, so the 3 per token of the plan are ~20 us (arithmetic from the measured median): the same order as the compute.
2. **Discrepancy.** tree4 with one release line was 10.10 us in run 1 and 13.25-16.03 us in run 2 (same algorithm). Between the runs, the barrier struct grew and the interleaving order changed. Unverified reading: when all 120 threads spin on one line, the release becomes a broadcast re-read that is sensitive to timing. The shapes where each release line has a single reader do not drift: 5 repeats within 0.2 us.
3. **Decision:** cortexd uses core2r (release down a core-aware tree). Design follow-up: cut barriers per token from 3 to 2. For example, every thread reduces h itself after the vote barrier, which needs a measurement of its read traffic.
4. **p99.9 is ~36 us for every shape:** OS/timer noise, so expect occasional spikes.
