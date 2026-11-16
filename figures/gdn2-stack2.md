# GDN2 Ascend chunk training A2 measurements

Baseline `fc3dc93c` → `618e0fef` (identical tested tree `c0b268a1`).

Seven dense BF16 shapes: forward 1.1744×, forward+backward 1.1275× geometric mean.

![A2 measurements](gdn2-a2-stack2.png)

[Samples, statistics, correctness, UB and profiler data](gdn2-stack2-measurements.json)

[Timing harness](gdn2-a2-bench.py): select the baseline/candidate checkout through `PYTHONPATH`, then run
`taskset -c 0-7 python gdn2-a2-bench.py --out result.json --warmups 3 --samples 10` in upstream → current → candidate → candidate → current → upstream order.

Physical NPU4 and CPU NUMA0 (cores0–7); 3 warmups and 10 synchronized samples per round, 20 per version/cell. Packed controls: 420 samples per version/cell.
Cold calls use three fresh processes and distinct empty Triton caches per route; vendor/OS caches remain warm.
Full GDN2 correctness passes. The broader suite retains two stochastic-rounding compiler failures reproduced on baseline; H100 remains unrun.
