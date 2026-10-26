# Stage 0b: toolchain + run path (2026-10-25 09:02)

Commands: `./build.sh` (S3 clang -march=knc, knc-cc -static, phi-isa-audit); `phi put build/card/hello /tmp/cortex/hello.new`, `mv`, `phi run ./hello`. Raw: `results/s00b-hello.log`.

| Check | Result |
|---|---|
| phi-isa-audit | 0 illegal, 0 suspect, 3 KNC vector |
| CPU | ran on CPU 66 (unpinned) |
| VPU add | 1.5 + 2.25 = 3.75 in lanes 0 and 15 |
| TSC | 1238.093 MHz over 1.000000 s = the core clock; 1 tick = 0.8077 ns |
| DCR | 0x0000169d: ch0 off/host, ch1 on/host, **ch2 off/host (free)**, ch3 on/card, ch4 on/card, ch5 off/host, ch6 off/host, ch7 off/card |

Channels 3 and 4 are still enabled card-owned from their (stopped) owners' sessions; not touched. Cortex takes channel 2 at Stage 4.
