# GDN-2 stack 1/4 performance on A2

These figures describe `ab836839` on Ascend 910B3 with CANN 9.2.0, PyTorch 2.10.0+cpu and torch_npu 2.10.0.post7.

The chunk figure compares the update with post-rebase `23e2e447`. For 3.2.2, the baseline uses the equivalent predicate rewrite that allows the old frontend to compile dense training. Each pair uses the same physical card and CPU-generated BF16 inputs. Device-event medians use 8 warmups and 16 samples.

The WY figure compares generic Triton and dedicated Ascend with the same BF16 inputs and FP32 cumulative gate in each run. Both routes execute on the same NPU. Host timings include allocation and dispatch. The pre-rebase WY comparison uses a separate set of CPU-generated inputs with matching hashes across revisions.

[Measurements](measurements.json) includes the raw samples, paired input hashes, stage profiles, and the pre-rebase comparisons.
