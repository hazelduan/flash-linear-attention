# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import gc
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
import triton

from tbe.common.platform import get_soc_spec, set_current_compile_soc_info

set_current_compile_soc_info(torch.npu.get_device_name(0))

import fla
from fla.utils.ascend_ub_manager import get_ub_manager

assert get_ub_manager().ub_capacity_bytes == int(get_soc_spec('UB_SIZE')) == 196608
from benchmarks.ops.registry import generate_inputs, get_op
from fla.ops.gdn2 import chunk_gdn2

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--samples', type=int, default=12)
parser.add_argument('--case')
parser.add_argument('--case-prefix')
parser.add_argument('--warmups', type=int, default=3)
args = parser.parse_args()
config = get_op('chunk_gdn2')
cases = [('production_B1_T2048_H16_D128', {'B': 1, 'T': 2048, 'H': 16, 'D': 128}, True)]
cases += [(name, shape, False) for name, shape in config.default_shapes.items()]
cases += [(f'packed_T145_D{dim}', {'B': 1, 'T': 145, 'H': 2, 'D': dim}, True) for dim in (32, 128)]
if args.case_prefix:
    cases = [case for case in cases if case[0].startswith(args.case_prefix)]
    assert cases, args.case_prefix
if args.case:
    cases = [case for case in cases if case[0] == args.case]
    assert cases, args.case
result = {
    'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
    'triton': triton.__version__, 'triton_ascend': importlib.metadata.version('triton-ascend'),
    'fla_path': fla.__file__, 'seed': 42, 'samples': args.samples, 'warmups': args.warmups,
    'method': 'synchronized host wall and NPU Events; autograd.grad for fwdbwd', 'results': [],
}
source_root = Path(fla.__file__).parent.parent
result['source_hashes'] = {
    str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (source_root / 'fla/ops/gdn2').rglob('*.py')
}

def summary(samples):
    deciles = statistics.quantiles(samples, n=10, method='inclusive')
    return {
        'median': statistics.median(samples), 'mean': statistics.mean(samples),
        'std': statistics.stdev(samples), 'min': min(samples),
        'p10': deciles[0], 'p90': deciles[8], 'samples': samples,
    }

print(json.dumps({k: v for k, v in result.items() if k not in ('source_hashes', 'results')}), flush=True)
for name, shape, fp32_gate in cases:
    torch.manual_seed(42)
    inputs = generate_inputs(config, **shape, dtype=torch.bfloat16, device='npu')
    if fp32_gate:
        inputs['g'] = inputs['g'].float().detach().requires_grad_(True)
    if name.startswith('packed_'):
        inputs['cu_seqlens'] = torch.tensor([0, 15, 80, 145], device='npu', dtype=torch.int64)
    do = torch.randn_like(inputs['v'])

    def fwd():
        return chunk_gdn2(**inputs, **config.extra_kwargs)[0]

    def fwdbwd():
        return torch.autograd.grad(fwd(), tuple(x for x in inputs.values() if x.requires_grad), do)

    for mode, fn in (('fwd', fwd), ('fwdbwd', fwdbwd)):
        print('warmup', name, mode, flush=True)
        for _ in range(args.warmups):
            fn()
        torch.npu.synchronize()
        wall, events = [], []
        for _ in range(args.samples):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            begin = time.perf_counter()
            start.record()
            fn()
            end.record()
            torch.npu.synchronize()
            wall.append((time.perf_counter() - begin) * 1000)
            events.append(start.elapsed_time(end))
        row = {'case': name, **shape, 'mode': mode, 'gate_dtype': str(inputs['g'].dtype),
               'wall_ms': summary(wall), 'event_ms': summary(events)}
        result['results'].append(row)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(name, mode, row['wall_ms']['median'], flush=True)
    del inputs, do
    gc.collect()
    torch.npu.empty_cache()
