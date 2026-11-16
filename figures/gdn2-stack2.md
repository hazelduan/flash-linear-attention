# GDN2 Ascend chunk training A2 measurements

Baseline `fc3dc93c` → `8a181658` (identical tested tree `2e787be1`).

Seven dense BF16 shapes: forward 1.1805×, forward+backward 1.1069× geometric mean.

![A2 measurements](gdn2-a2-stack2.png)

[Samples, statistics, correctness, UB and profiler data](gdn2-stack2-measurements.json)

[Timing harness](gdn2-a2-bench.py): select the baseline/candidate checkout through `PYTHONPATH`, then run
`python gdn2-a2-bench.py --out result.json --warmups 3 --samples 10` in base → candidate → candidate → base order.

Primary ABBA: 3 warmups and 10 synchronized samples per round, 20 per version/cell. Packed controls: 420 samples per version/cell.
Cold calls use three fresh processes and distinct empty Triton caches per route; vendor/OS caches remain warm.
Full GDN2 correctness passes. The broader suite retains two stochastic-rounding compiler failures reproduced on baseline; H100 remains unrun.
