# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend dispatch and leaf-kernel coverage for GDN-2."""


import pytest
import torch

from fla.utils import IS_NPU, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason="Ascend NPU required")

_BT = 64
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _sequence_spans(B: int, T: int, lengths: tuple[int, ...] | None) -> list[tuple[int, int, int]]:
    if lengths is None:
        return [(i_b, 0, T) for i_b in range(B)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return [(0, offsets[i], offsets[i + 1]) for i in range(len(lengths))]


def _make_wy_inputs(
    *,
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    spans: list[tuple[int, int, int]],
    has_q: bool,
) -> tuple[torch.Tensor | None, ...]:
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.25 if has_q else None
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.25
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.25
    b = torch.rand(B, T, H, K, dtype=dtype, device=device) * 0.5
    w_gate = torch.rand(B, T, H, V, dtype=dtype, device=device) * 0.5
    gk = torch.empty(B, T, H, K, dtype=torch.float32, device=device)
    A = torch.zeros(B, T, H, _BT, dtype=torch.float32, device=device)

    for i_b, seq_start, seq_end in spans:
        for chunk_start in range(seq_start, seq_end, _BT):
            chunk_end = min(chunk_start + _BT, seq_end)
            chunk_len = chunk_end - chunk_start
            increments = -(torch.rand(chunk_len, H, K, device=device) * 0.02 + 0.01)
            gk[i_b, chunk_start:chunk_end] = increments.cumsum(0)

            eye = torch.eye(_BT, dtype=torch.float32, device=device).expand(H, -1, -1)
            noise = torch.randn(H, _BT, _BT, dtype=torch.float32, device=device) * 0.01
            block = eye + torch.tril(noise, diagonal=-1)
            A[i_b, chunk_start:chunk_end] = block[:, :chunk_len].permute(1, 0, 2)

    return q, k, v, b, w_gate, A.to(dtype), gk.to(dtype)


def _wy_reference(
    *,
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    spans: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    gk_exp = torch.exp2(gk.float())
    qg = (q.float() * gk_exp).to(q.dtype) if q is not None else None

    for i_b, seq_start, seq_end in spans:
        for chunk_start in range(seq_start, seq_end, _BT):
            chunk_end = min(chunk_start + _BT, seq_end)
            chunk_len = chunk_end - chunk_start
            A_chunk = A[i_b, chunk_start:chunk_end].permute(1, 0, 2).float()[..., :chunk_len]

            vb = (v[i_b, chunk_start:chunk_end].float() *
                  w_gate[i_b, chunk_start:chunk_end].float()).to(v.dtype)
            u_chunk = torch.matmul(A_chunk, vb.permute(1, 0, 2).float())
            u[i_b, chunk_start:chunk_end] = u_chunk.permute(1, 0, 2).to(v.dtype)

            kb = (k[i_b, chunk_start:chunk_end].float() *
                  b[i_b, chunk_start:chunk_end].float() *
                  gk_exp[i_b, chunk_start:chunk_end]).to(k.dtype)
            w_chunk = torch.matmul(A_chunk, kb.permute(1, 0, 2).float())
            w[i_b, chunk_start:chunk_end] = w_chunk.permute(1, 0, 2).to(k.dtype)

            g_last = gk[i_b, chunk_end - 1].float()
            decay = torch.exp2(g_last[None, :, :] - gk[i_b, chunk_start:chunk_end].float())
            kg[i_b, chunk_start:chunk_end] = (k[i_b, chunk_start:chunk_end].float() * decay).to(k.dtype)

    return w, u, qg, kg


@pytest.mark.parametrize(
    ("layout", "dtype", "K", "V", "has_q"),
    [
        pytest.param("dense", torch.float16, 48, 32, False, id="dense-fp16-k48-v32-no-q"),
        pytest.param("dense", torch.float32, 32, 48, True, id="dense-fp32-k32-v48-q"),
        pytest.param("dense", torch.bfloat16, 256, 256, True, id="dense-bf16-k256-v256-q"),
        pytest.param("varlen", torch.bfloat16, 48, 32, False, id="varlen-bf16-k48-v32-no-q"),
        pytest.param("varlen", torch.float16, 32, 48, True, id="varlen-fp16-k32-v48-q"),
    ],
)
def test_recompute_w_u_fwd_gdn2_npu_leaf(layout, dtype, K, V, has_q):
    """The Ascend WY leaf must match its channel-wise Torch definition."""
    from fla.ops.gdn2.backends.triton_ascend.wy_fast import recompute_w_u_fwd_gdn2_npu

    lengths = None if layout == "dense" else (15, 65, 65)
    B = 2 if layout == "dense" and K < 256 else 1
    H = 2 if K < 256 else 1
    T = 65 if lengths is None else sum(lengths)
    spans = _sequence_spans(B, T, lengths)
    q, k, v, b, w_gate, A, gk = _make_wy_inputs(
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        dtype=dtype,
        spans=spans,
        has_q=has_q,
    )
    cu_seqlens = None
    if lengths is not None:
        cu_seqlens = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), device=device)

    expected = _wy_reference(q=q, k=k, v=v, b=b, w_gate=w_gate, A=A, gk=gk, spans=spans)
    actual = recompute_w_u_fwd_gdn2_npu(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=A,
        q=q,
        gk=gk,
        cu_seqlens=cu_seqlens,
    )

    for name, ref, tri in zip(("w", "u", "qg", "kg"), expected, actual):
        if ref is None:
            assert tri is None
        else:
            assert tri is not None
            assert_close(name, ref, tri, 0.005)


def _wy_verifier_args(
    *,
    dtype: torch.dtype = torch.float16,
    B: int = 1,
    K: int = 32,
    V: int = 32,
    chunk_size: int = _BT,
) -> dict[str, torch.Tensor]:
    k = torch.zeros(B, 1, 1, K, dtype=dtype, device=device)
    v = torch.zeros(B, 1, 1, V, dtype=dtype, device=device)
    return {
        "k": k,
        "v": v,
        "b": torch.zeros_like(k),
        "w_gate": torch.zeros_like(v),
        "A": torch.zeros(B, 1, 1, chunk_size, dtype=dtype, device=device),
        "q": torch.zeros_like(k),
        "gk": torch.zeros_like(k),
    }


def test_recompute_w_u_fwd_gdn2_verifier_accepts_partitioned_v():
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    accepted, reason = TritonAscendGDN2Backend().recompute_w_u_fwd_gdn2_verifier(
        **_wy_verifier_args(V=257),
    )
    assert accepted
    assert reason is None


@pytest.mark.parametrize(
    ("case", "reason_fragment"),
    [
        pytest.param("chunk-size", "chunk_size=64", id="chunk-size"),
        pytest.param("k-zero", "1 <= K <= 256", id="k-zero"),
        pytest.param("k-oversized", "1 <= K <= 256", id="k-oversized"),
        pytest.param("v-zero", "V >= 1", id="v-zero"),
        pytest.param("packed-batch", "batch size 1", id="packed-batch"),
        pytest.param("gk-none", "requires gk", id="gk-none"),
        pytest.param("index-dtype", "int32 or int64", id="index-dtype"),
        pytest.param("cu-seqlens-shape", "cu_seqlens to be rank 1", id="cu-seqlens-shape"),
        pytest.param("chunk-indices-shape", "chunk_indices shape [NT, 2]", id="chunk-indices-shape"),
        pytest.param("shape", "shape", id="shape"),
        pytest.param("device", "same NPU device", id="device"),
        pytest.param("dtype", "unsupported dtype", id="dtype"),
    ],
)
def test_recompute_w_u_fwd_gdn2_verifier_rejects_invalid_contract(case, reason_fragment):
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    if case == "k-zero":
        kwargs = _wy_verifier_args(K=0)
    elif case == "k-oversized":
        kwargs = _wy_verifier_args(K=257)
    elif case == "v-zero":
        kwargs = _wy_verifier_args(V=0)
    elif case == "packed-batch":
        kwargs = _wy_verifier_args(B=2)
        kwargs["cu_seqlens"] = torch.tensor([0, 1], dtype=torch.int64, device=device)
    elif case == "gk-none":
        kwargs = _wy_verifier_args()
        kwargs["gk"] = None
    else:
        kwargs = _wy_verifier_args(chunk_size=32 if case == "chunk-size" else _BT)
        if case == "index-dtype":
            kwargs["cu_seqlens"] = torch.tensor([0, 1], dtype=torch.float32, device=device)
        elif case == "cu-seqlens-shape":
            kwargs["cu_seqlens"] = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        elif case == "chunk-indices-shape":
            kwargs["chunk_indices"] = torch.tensor([0, 0], dtype=torch.int64, device=device)
        elif case == "shape":
            kwargs["b"] = torch.zeros(1, 1, 1, 31, dtype=torch.float16, device=device)
        elif case == "device":
            kwargs["q"] = kwargs["q"].cpu()
        elif case == "dtype":
            kwargs["b"] = kwargs["b"].to(torch.int32)

    accepted, reason = TritonAscendGDN2Backend().recompute_w_u_fwd_gdn2_verifier(**kwargs)
    assert not accepted
    assert reason is not None
    assert reason_fragment in reason
