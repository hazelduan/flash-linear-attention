# GDN-2 Ascend intra backward, stack 2/4

Baseline main `2705ab8ee9a9086bcfa6d079858e4281e01215a1` includes merged #1606. Candidate `244262e118248358fa7876f6cb46afc9cf8e0874` adds the intra-backward backend and a measured UB-budget bound. The performance result is a regression, so #1739 remains draft.

Same Ascend 910B3/A2, CANN 9.1.0, PyTorch 2.10.0, torch_npu 2.10.0.post7 and Triton-Ascend 3.6.0. Whole-operator timings use physical device 4, four fresh processes in base/candidate/candidate/base order, three warmups and ten synchronized samples per cell per round. The combined twenty samples per version/cell retain median, mean, sample standard deviation, min, p10, p90 and raw wall/NPU-event values. Seven dense BF16 shapes plus one FP32-gate and two packed controls are present. Seed 42; allocation/dispatch included; compilation excluded. Two-sample prewarm records are excluded.

Independent leaf inputs are captured from actual GDN2 training calls. The unchanged Default Triton body is AST-checked against the baseline, native dispatch is observed, and every returned gradient is compared before timing. Six workloads use three warmups and thirty paired samples per route, reversing order halfway. Representative and 32K PipeUtilization/MemoryUB reports are summarized in the dataset.

Cold leaf calls use physical device 7, three fresh processes and initially empty separate Triton caches per route, alternating order. The clock starts after imports and input placement. It includes compilation, autotuning, allocation and launch, and is separate from whole-operator startup. Compiler subprocess times and counts are retained. Vendor/OS caches are not cleared.

Strict GDN2 gates pass 103/60 skips on the baseline and 115/60 skips on the candidate. Broad modules/ops-utils/backend/public-API coverage reports 1043 passed, 544 skipped and 2 failed; both stochastic-rounding compiler failures reproduce on the unchanged baseline. No numerical-tolerance warning is present. H100 dispatch is blocked by GitHub repository permissions and remains unrun.

Twelve fresh-cache kernel compilations cover FP16/BF16/FP32, safe and ordinary gates, and BK16/BK32 at K=256. Post-PlanMemory UB address high-water is at most 70144 bytes / 124160 bytes respectively. Predicate buffers use a conservative one-byte-per-element extent. The host planner budgets 72 FP32 tile equivalents with 25% headroom and accepts 96 KiB or more; actual 192 KiB uses BK32. Raw traces, compiler IR, commands, per-file source hashes and rejected scheduling diagnostics are preserved in the local validation archive.
