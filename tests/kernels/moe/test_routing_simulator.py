#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for the token-to-expert routing simulator.

This script demonstrates how to use the routing simulator to test
different routing strategies and analyze their performance, including
integration tests with FusedMoEFactory layer.
"""

import tempfile

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    DistributionBasedRouting,
    RoutingSimulator,
)


@pytest.fixture
def device():
    """Fixture to provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("num_tokens", [1, 16, 256])
@pytest.mark.parametrize("hidden_size", [64, 1024])
@pytest.mark.parametrize("num_experts", [16, 128])
@pytest.mark.parametrize("top_k", [1, 4])
def test_basic_functionality(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device,
):
    """Test basic functionality of the routing simulator."""
    # Test each routing strategy
    strategies = RoutingSimulator.get_available_strategies()

    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    for strategy in strategies:
        # Simulate routing
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=strategy,
            top_k=top_k,
        )

        # Check output shapes
        assert topk_weights.shape == (
            num_tokens,
            top_k,
        ), f"Wrong weights shape for {strategy}"
        assert topk_ids.shape == (
            num_tokens,
            top_k,
        ), f"Wrong ids shape for {strategy}"

        # Check that expert IDs are valid
        assert topk_ids.min() >= 0, f"Invalid expert ID (negative) for {strategy}"
        assert topk_ids.max() < num_experts, (
            f"Invalid expert ID (too large) for {strategy}"
        )


def test_routing_strategy_integration(monkeypatch, device):
    """Test that the routing strategy environment variable works with
    FusedMoEFactory."""
    pytest.importorskip("vllm.model_executor.layers.fused_moe.layer")

    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

    # Test parameters
    num_tokens = 32
    hidden_size = 16
    num_experts = 4
    top_k = 2

    # Create test data
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Test different routing strategies
    strategies = RoutingSimulator.get_available_strategies()

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        for strategy in strategies:
            fused_moe = FusedMoEFactory(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=0,
                use_grouped_topk=False,
                renormalize=True,
                prefix=strategy,
            )

            # Set environment variable
            env_name = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
            monkeypatch.setenv(env_name, strategy)

            # Temporarily override the envs lookup so the router factory
            # reads the monkeypatched value instead of the module-load-time
            # default. Use monkeypatch.setitem so the original lambda is
            # restored automatically at teardown.
            monkeypatch.setitem(
                envs.environment_variables,
                env_name,
                lambda s=strategy: s,
            )

            # Test the select_experts method
            topk_weights, topk_ids = fused_moe.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            # Verify output shapes
            assert topk_weights.shape == (num_tokens, top_k), (
                f"Wrong weights shape for {strategy}"
            )
            assert topk_ids.shape == (num_tokens, top_k), (
                f"Wrong ids shape for {strategy}"
            )

            # Verify expert IDs are valid
            assert topk_ids.min() >= 0, f"Invalid expert ID (negative) for {strategy}"
            assert topk_ids.max() < num_experts, (
                f"Invalid expert ID (too large) for {strategy}"
            )


def test_distribution_based_routing_with_custom_strategy(monkeypatch):
    """Test registering and using DistributionBasedRouting with custom
    parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    monkeypatch.setattr(
        RoutingSimulator,
        "_routing_strategies",
        dict(RoutingSimulator._routing_strategies),
    )
    custom_strategy = DistributionBasedRouting(distribution="normal", mean=2.0, std=0.5)
    RoutingSimulator.register_strategy("custom_normal", custom_strategy)

    # Test data
    num_tokens = 60
    hidden_size = 48
    num_experts = 6
    top_k = 3

    hidden_states = torch.randn(num_tokens, hidden_size, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    # Use the custom strategy
    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name="custom_normal",
        top_k=top_k,
    )

    # Check output shapes
    assert topk_weights.shape == (num_tokens, top_k)
    assert topk_ids.shape == (num_tokens, top_k)

    # Check that expert IDs are valid
    assert topk_ids.min() >= 0
    assert topk_ids.max() < num_experts


def test_instance_compatibility():
    """Test that static methods work correctly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test static method directly
    hidden_states = torch.randn(10, 8, device=device)
    router_logits = torch.randn(10, 4, device=device)

    topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name="uniform_random",
        top_k=2,
    )

    assert topk_weights.shape == (10, 2)
    assert topk_ids.shape == (10, 2)


def _normalized_counts(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).float()
    return counts / counts.sum()


def _has_duplicate_row(topk_ids: torch.Tensor) -> torch.Tensor:
    sorted_ids = topk_ids.sort(dim=-1).values
    return (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(dim=-1)


def _participation_ratio(topk_ids: torch.Tensor, num_experts: int) -> float:
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).float()
    return (counts.sum() ** 2 / (counts**2).sum()).item()


@pytest.mark.parametrize("strategy_name", ["uniform_random", "normal_routing"])
@pytest.mark.parametrize("num_experts,top_k", [(16, 1), (16, 16), (256, 8), (4096, 8)])
def test_deepep_v2_strategies_emit_distinct_experts(
    strategy_name: str,
    num_experts: int,
    top_k: int,
    device,
):
    """DeepEP v2 requires distinct top-k expert IDs for every token."""
    num_tokens = 64
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, 64, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    _topk_weights, topk_ids = RoutingSimulator.simulate_routing(
        hidden_states=hidden_states,
        router_logits=router_logits,
        strategy_name=strategy_name,
        top_k=top_k,
    )

    assert topk_ids.shape == (num_tokens, top_k)
    assert topk_ids.min() >= 0
    assert topk_ids.max() < num_experts
    duplicated = _has_duplicate_row(topk_ids)
    if duplicated.any():
        token_idx = int(duplicated.nonzero()[0])
        raise AssertionError(
            f"{strategy_name} routed token {token_idx} to duplicate experts: "
            f"{topk_ids[token_idx].tolist()}"
        )


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_strategies_reject_top_k_above_num_experts(distribution: str, device):
    strategy = DistributionBasedRouting(distribution=distribution)
    with pytest.raises(ValueError, match="cannot exceed num_experts"):
        strategy.route_tokens(
            hidden_states=torch.randn(4, 64, device=device),
            router_logits=torch.randn(4, 8, device=device),
            top_k=16,
        )


def test_normal_routing_preserves_load_skew(device):
    num_tokens, num_experts, top_k = 4096, 256, 8
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, 64, device=device)
    router_logits = torch.randn(num_tokens, num_experts, device=device)

    def route(strategy_name: str) -> float:
        _weights, ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=strategy_name,
            top_k=top_k,
        )
        return _participation_ratio(ids, num_experts)

    skewed = route("normal_routing")
    uniform = route("uniform_random")

    assert skewed < 215.0, (
        f"normal_routing participation ratio {skewed:.1f} is too high; "
        "the normal-routing load skew was lost"
    )
    assert uniform > 250.0


@pytest.mark.parametrize(
    "num_experts,mean,std",
    [
        (256, 0.0, 1.0),
        (256, 2.0, 0.5),
        (256, 0.0, 0.3),
        (1024, 0.0, 0.25),
        (4096, 0.0, 1.0),
    ],
)
def test_normal_log_marginal_is_finite_and_ordered(
    num_experts: int, mean: float, std: float, device
):
    strategy = DistributionBasedRouting(distribution="normal", mean=mean, std=std)
    log_probs = strategy._expert_log_marginal(num_experts, torch.device(device))

    assert log_probs.shape == (num_experts,)
    assert log_probs.isfinite().all()
    assert abs(torch.logsumexp(log_probs.double(), dim=0).item()) < 1e-6
    assert log_probs.min().item() > -700.0

    mode = int(log_probs.argmax())
    if mode > 1:
        left = log_probs[: mode + 1]
        assert (left[1:] - left[:-1] > 0).all()


def test_normal_routing_matches_the_binned_normal_marginal(device):
    num_experts, num_tokens, mean, std = 256, 200_000, 2.0, 0.5
    strategy = DistributionBasedRouting(distribution="normal", mean=mean, std=std)
    torch.manual_seed(0)
    ids = strategy._sample_expert_ids(
        num_tokens, num_experts, 1, torch.device(device), torch.long
    )
    got = _normalized_counts(ids, num_experts)

    torch.manual_seed(0)
    reference = strategy._sample_continuous_distribution(
        num_tokens, 1, torch.device(device)
    )
    reference = (torch.sigmoid(reference) * num_experts).long()
    reference.clamp_(0, num_experts - 1)
    want = _normalized_counts(reference, num_experts)

    tv = 0.5 * (got - want).abs().sum().item()
    assert tv < 0.02


def test_expert_log_marginal_is_cached(device):
    strategy = DistributionBasedRouting(distribution="normal", mean=0.0, std=1.0)
    dev = torch.device(device)

    first = strategy._expert_log_marginal(256, dev)
    second = strategy._expert_log_marginal(256, dev)
    assert first is second

    other = strategy._expert_log_marginal(128, dev)
    assert other is not first
    assert other.shape == (128,)

    narrow = DistributionBasedRouting(distribution="normal", mean=0.0, std=0.25)
    assert not torch.allclose(narrow._expert_log_marginal(256, dev), first)
