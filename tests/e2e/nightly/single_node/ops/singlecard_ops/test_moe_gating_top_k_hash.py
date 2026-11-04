import pytest
import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

NUM_EXPERTS = 64
TOP_K = 6
ROUTED_SCALING_FACTOR = 1.5
EPS = 1e-20


def sqrtsoftplus_topk_golden(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden result of the fused op: scores = sqrt(softplus(x)), top-k,
    renormalize over the selected experts and scale."""
    scores = torch.sqrt(F.softplus(router_logits))
    top_scores, top_ids = torch.topk(scores, TOP_K, dim=-1)
    weights = top_scores / (top_scores.sum(-1, keepdim=True) + EPS) * ROUTED_SCALING_FACTOR
    return weights, top_ids


def run_op(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=router_logits.npu(),
        k=TOP_K,
        bias=torch.zeros(NUM_EXPERTS, dtype=router_logits.dtype).npu(),
        input_ids=None,
        tid2eid=None,
        k_group=1,
        group_count=1,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        eps=EPS,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )
    return weights.cpu(), ids.cpu()


@pytest.mark.parametrize(
    "logits",
    [
        # Real-world signature: draft-model router logits with absmax ~127,
        # where 5 elements exceed the fp32 exp limit (~88.7).
        torch.tensor([
            [126.935, 121.0, 71.2, 63.6, 57.9, 53.8, 26.5, -12.3],
            [122.8, 118.4, 69.9, 62.1, 56.4, 52.2, 25.1, -14.7],
            [124.6, 119.8, 70.5, 63.0, 57.1, 53.0, 25.8, -13.5],
            [125.2, 120.3, 70.9, 63.3, 57.4, 53.3, 26.1, -12.9],
            [123.7, 119.1, 70.1, 62.5, 56.7, 52.6, 25.4, -14.1],
        ]),
        # Extreme synthetic values well beyond the fp32 exp limit.
        torch.tensor([
            [1000.0, 500.0, 127.0, 100.0, 89.0, 30.0, 21.0, 20.5],
            [-500.0, -100.0, 88.72, 89.0, 121.0, 71.2, 63.6, 57.9],
        ]),
    ],
    ids=["draft-scale-logits", "extreme-logits"],
)
@torch.inference_mode()
def test_sqrtsoftplus_large_logits_produce_finite_output(logits: torch.Tensor):
    """Router logits beyond the fp32 exp limit must not yield NaN weights."""
    torch.manual_seed(45)
    pad = torch.randn(logits.shape[0], NUM_EXPERTS - logits.shape[1], dtype=torch.float32) * 5
    router_logits = torch.cat([logits, pad], dim=1).to(torch.float32)
    assert bool((router_logits > 88.7).any()), "test requires overflow-scale logits"

    weights, ids = run_op(router_logits)

    assert not bool(torch.isnan(weights).any()), (
        f"NaN in top-k weights for logits absmax={float(router_logits.abs().max())}"
    )
    expected_weights, expected_ids = sqrtsoftplus_topk_golden(router_logits)
    torch.testing.assert_close(ids.to(torch.int64), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-7)


@torch.inference_mode()
def test_sqrtsoftplus_normal_range_matches_golden():
    """Logits in the normal range keep matching the golden reference."""
    torch.manual_seed(45)
    router_logits = torch.rand(8, NUM_EXPERTS, dtype=torch.float32) * 50 - 38
    assert not bool((router_logits > 20).any()), "this case must stay below the linear threshold"

    weights, ids = run_op(router_logits)

    expected_weights, expected_ids = sqrtsoftplus_topk_golden(router_logits)
    torch.testing.assert_close(ids.to(torch.int64), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-7)


@torch.inference_mode()
def test_sqrtsoftplus_bf16_large_logits_produce_finite_output():
    """bf16 router logits beyond the fp32 exp limit must not yield NaN either."""
    torch.manual_seed(45)
    base = torch.tensor([
        [127.0, 121.0, 71.2, 63.6, 57.9, 53.8, 26.5, -12.3],
        [122.8, 118.4, 69.9, 62.1, 56.4, 52.2, 25.1, -14.7],
    ])
    pad = torch.randn(2, NUM_EXPERTS - base.shape[1], dtype=torch.float32) * 5
    router_logits = torch.cat([base, pad], dim=1).to(torch.bfloat16)
    assert bool((router_logits.float() > 88.7).any()), "test requires overflow-scale logits"

    weights, ids = run_op(router_logits)

    assert not bool(torch.isnan(weights).any()), "NaN in top-k weights for bf16 logits"
    expected_weights, expected_ids = sqrtsoftplus_topk_golden(router_logits.float())
    # bf16 output is cast back with round-to-nearest, keep a coarse tolerance.
    torch.testing.assert_close(ids.to(torch.int64), expected_ids)
    torch.testing.assert_close(weights.float(), expected_weights, rtol=1e-2, atol=1e-3)
