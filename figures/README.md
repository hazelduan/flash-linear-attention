# GDN-2 stack 1/4 A2 figures

The chunk-training figure retains the original complete first-stack comparison from b0de6fc8 to ba004dcc. It includes grouped-intra contiguous-row writeback and WY recomputation on the same Ascend 910B3 for each compiler. The software stack uses CANN 9.1.0 and PyTorch 2.10.0.

[Original chunk measurements and environment](https://github.com/hazelduan/flash-linear-attention/blob/b8313a3856a69caf1d9be442e3d4e3f60b8ffb88/figures/measurements.json) record the raw samples and exact protocol. The figure names both measured commits explicitly.

The WY figure and [measurements.json](measurements.json) describe ab836839 on CANN 9.2.0, including the Default-versus-Ascend WY comparisons and the latest incremental update from 23e2e447. Default and Ascend execute the same inputs on the same NPU within each pair.
