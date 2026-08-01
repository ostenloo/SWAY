"""A1 (second half) — the C8 family-disjointness assertion.

As a *reward*, a champion sharing a model family with the finetune base is not
merely a blind spot that goes unnoticed — it becomes a gradient the policy farms.
So this is asserted at construction, and an unrecognised model is a FAILURE rather
than a pass.
"""

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.reward.fidelity_reward import (
    FamilyDisjointnessError, assert_family_disjoint, model_family,
)

QWEN_BASE = "Qwen/Qwen2.5-14B-Instruct"


def test_family_inference_covers_the_configured_models():
    assert model_family(QWEN_BASE) == "qwen"
    assert model_family("qwen2.5:14b-instruct-q4_K_M") == "qwen"
    assert model_family("command-r7b:latest") == "cohere"
    assert model_family("glm4:9b") == "zhipu"
    assert model_family("gemma2:9b") == "gemma"
    assert model_family("llama3.1:8b-instruct-q4_K_M") == "llama"


def test_configured_champions_are_disjoint_from_the_qwen_base():
    """The shipped config: command-r7b (cohere) + glm4 (zhipu) vs a Qwen base."""
    resolved = assert_family_disjoint(
        QWEN_BASE, engine="command-r7b:latest", delivery="glm4:9b",
    )
    assert resolved == {"base": "qwen", "engine": "cohere", "delivery": "zhipu"}


def test_same_family_champion_is_rejected():
    with pytest.raises(FamilyDisjointnessError, match="C8 BREACH"):
        assert_family_disjoint(QWEN_BASE, engine="qwen2.5:7b", delivery="glm4:9b")


def test_same_family_delivery_is_rejected_too():
    with pytest.raises(FamilyDisjointnessError, match="delivery"):
        assert_family_disjoint(QWEN_BASE, engine="command-r7b:latest", delivery="qwq:32b")


def test_unknown_model_is_a_failure_not_a_pass():
    """Silently admitting an unrecognised tag is how a same-family reward slips in."""
    with pytest.raises(FamilyDisjointnessError, match="cannot determine"):
        assert_family_disjoint(QWEN_BASE, engine="some-new-model:8b", delivery="glm4:9b")


def test_unknown_base_is_a_failure():
    with pytest.raises(FamilyDisjointnessError, match="base_model"):
        assert_family_disjoint("acme/mystery-13b", engine="command-r7b:latest",
                               delivery="glm4:9b")


def test_ministral_fallback_base_still_disjoint():
    """R5's 8B fallback (Ministral) keeps both champions legal."""
    resolved = assert_family_disjoint(
        "mistralai/Ministral-8B-Instruct", engine="command-r7b:latest", delivery="glm4:9b",
    )
    assert resolved["base"] == "mistral"


def test_reward_construction_asserts_c8():
    """The rail lives on the construction path, not only in a helper."""
    from grpo.reward.backends import build_champion_backends

    with pytest.raises(FamilyDisjointnessError):
        build_champion_backends(
            engine_model="qwen2.5:7b",          # same family as the base
            delivery_model="glm4:9b",
            base_url="http://localhost:11434/v1",
            base_model=QWEN_BASE,
        )


def test_engine_and_delivery_are_separate_cores():
    """R4: axis-local blind spots require two DISTINCT models, not one shared core."""
    from grpo.reward.backends import build_champion_backends

    b = build_champion_backends(
        engine_model="command-r7b:latest", delivery_model="glm4:9b",
        base_url="http://localhost:11434/v1", base_model=QWEN_BASE,
    )
    assert b.delivery.core is not b.engine.core
    assert not hasattr(b, "realism")
