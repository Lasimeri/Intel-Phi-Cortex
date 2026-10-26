# Stage 0a: card facts (2026-10-25 09:00 host time, 13:57 card clock)

Command: `phi run sh -c '...'` (nproc, /proc/meminfo, /proc/swaps, /proc/loadavg, /proc/PID/stat field 39, ps, /tmp/phi-cdma*.lock). Raw: `results/s00a-facts.log`.

| Fact | Value |
|---|---|
| CPUs | nproc 244 |
| GDDR | MemTotal 15,789,924 kB; MemFree 11,130,884 kB |
| Huge pages | 2052 total, 2052 free (2048 kB each) |
| Swap | none active |
| User processes | init, syslogd, klogd, dropbear, phi-agent (CPU 17), login sh; nothing else |
| Block pollers | phiblk0-poll CPU 239, phiblk1-poll CPU 238 (core 59), state D; 12 ticks of CPU each over 5 s (2.4 % if HZ 100, less at a higher HZ) |
| Load average | 2.11 (the two D-state pollers count toward it) |
| DMA locks on card | /tmp/phi-cdma3.lock, /tmp/phi-cdma4.lock (other daemons' channels); channel 2 has no lock file |
| Region part chosen | 1344 MiB, 128 MiB (first 64 B were zero; neighbours from protocol headers) |

Result: the card is free for Cortex. The core-59 column threads use CPUs 237 and 240, beside pollers that are mostly asleep. Channel 2's DCR bits are read in Stage 0b (card side, read-only /dev/mem). `phictl regs` would open VFIO, and every VFIO open resets the bus, killing the running card.
