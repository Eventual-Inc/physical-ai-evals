"""Policy adapter tests with injected lightweight backends."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from physical_ai_evals.policy import (
    OPENVLA_CHECKPOINTS,
    OpenVLAPolicy,
    PolicySpec,
    VLAJEPAPolicy,
    openvla,
    vla_jepa,
)


class _Inputs(dict):
    def to(self, device, dtype):
        self["device"] = device
        self["dtype"] = dtype
        return self


class _Processor:
    def __init__(self):
        self.prompt = None
        self.image = None

    def __call__(self, prompt, image):
        self.prompt = prompt
        self.image = image
        return _Inputs(input_ids=[[1]])


class _OpenVLAModel:
    norm_stats = {"libero_goal": {}}

    def __init__(self, action=None):
        self.action = (
            np.asarray(action, dtype=np.float32)
            if action is not None
            else np.arange(7, dtype=np.float32)
        )
        self.call = None

    def predict_action(self, **kwargs):
        self.call = kwargs
        return self.action


def test_openvla_prompt_action_and_gripper_contract():
    processor = _Processor()
    model = _OpenVLAModel([0, 0, 0, 0, 0, 0, 0.996])
    policy = OpenVLAPolicy(
        "fixture",
        unnorm_key="libero_goal",
        device="cpu",
        center_crop=False,
        _model=model,
        _processor=processor,
    )

    policy.reset("put the bowl on the plate")
    action = policy.act({"image": np.zeros((8, 8, 3), dtype=np.uint8)})

    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert action[-1] == -1.0
    assert "put the bowl on the plate" in processor.prompt
    assert model.call["unnorm_key"] == "libero_goal"
    assert model.call["do_sample"] is False


def test_openvla_center_crop_preserves_shape():
    pytest.importorskip("PIL")
    processor = _Processor()
    policy = OpenVLAPolicy(
        "fixture",
        unnorm_key="libero_goal",
        device="cpu",
        _model=_OpenVLAModel(),
        _processor=processor,
    )
    policy.reset("task")
    policy.act({"image": np.full((20, 24, 3), 127, dtype=np.uint8)})
    assert np.asarray(processor.image).shape == (20, 24, 3)


def test_openvla_factory_is_suite_specific_and_revision_pinned():
    for suite, (model_id, revision, unnorm_key) in OPENVLA_CHECKPOINTS.items():
        spec = openvla(suite)
        assert spec.policy_id == model_id
        assert spec.revision == revision
        assert spec.metadata["unnorm_key"] == unnorm_key == suite

    with pytest.raises(ValueError, match="requires one of"):
        openvla("libero_90")
    with pytest.raises(ValueError, match="immutable revision"):
        openvla("libero_goal", model_id="research/custom")
    with pytest.raises(ValueError, match="does not match"):
        openvla("libero_goal", unnorm_key="libero_spatial")


class _LeRobotPolicy:
    def __init__(self):
        import torch

        self.config = object()
        self.resets = 0
        self.batches = []
        self.action = torch.tensor(
            [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0]],
            dtype=torch.float32,
        )

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        self.batches.append(batch)
        return self.action


def _vla_jepa(fake):
    with patch.object(VLAJEPAPolicy, "_load"):
        policy = VLAJEPAPolicy("fixture", device="cpu")
    policy.model = fake
    policy.preprocessor = lambda batch: batch
    policy.postprocessor = lambda action: action
    return policy


def _observation():
    return {
        "image": np.full((8, 8, 3), 255, dtype=np.uint8),
        "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "instruction": "",
    }


def test_vla_jepa_batch_contract_and_negative_stride_images():
    torch = pytest.importorskip("torch")
    fake = _LeRobotPolicy()
    policy = _vla_jepa(fake)
    policy.reset("pick up the cup")

    observation = _observation()
    observation["image"] = observation["image"][::-1, ::-1]
    observation["wrist_image"] = observation["wrist_image"][::-1, ::-1]
    action = policy.act(observation)

    assert fake.resets == 1
    assert action.shape == (7,)
    batch = fake.batches[0]
    assert set(batch) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "task",
    }
    assert batch["task"] == "pick up the cup"
    assert isinstance(batch["observation.images.image"], torch.Tensor)
    assert batch["observation.images.image"].shape == (1, 3, 8, 8)
    assert batch["observation.state"].shape == (1, 8)


def test_vla_jepa_requires_wrist_and_records_dependency_revisions():
    pytest.importorskip("torch")
    policy = _vla_jepa(_LeRobotPolicy())
    policy.reset("task")
    with pytest.raises(ValueError, match="wrist"):
        policy.act({"image": np.zeros((8, 8, 3), dtype=np.uint8)})

    spec = vla_jepa()
    assert "@" in spec.metadata["qwen3_vl"]
    assert "@" in spec.metadata["vjepa2"]


def test_policy_spec_accepts_structural_user_policy_and_rejects_unsafe_identity():
    class UserPolicy:
        action_dim = 7
        control_mode = "relative"

        def reset(self, instruction):
            pass

        def act(self, observation):
            return np.zeros(7, dtype=np.float32)

        def close(self):
            pass

    spec = PolicySpec(UserPolicy, "me/policy", "source-commit-123")
    assert isinstance(spec.factory(), UserPolicy)

    with pytest.raises(ValueError, match="revision"):
        PolicySpec(UserPolicy, "me/policy", "")
    with pytest.raises(ValueError, match="relative"):
        PolicySpec(UserPolicy, "me/policy", "revision", control_mode="absolute")
