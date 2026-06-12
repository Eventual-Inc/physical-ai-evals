"""Policy adapter tests — OpenVLA + VLA-JEPA plumbing via injected fake backends.

No GPU / weights / torch / transformers / lerobot needed: each adapter takes an injected
backend (``_vla``/``_processor``, ``_policy``) so we can verify prompt construction, batch
assembly, action post-processing (shape/dtype/clip-to-7), control_mode, and the required
chunk-reset — the parts that don't need real inference. Real-weight inference is exercised in
the GPU env (Modal), not here.
"""

from __future__ import annotations

import numpy as np

from harness.policies.openvla import LIBERO_CHECKPOINTS, OpenVLAPolicy
from harness.policies.vla_jepa import VLAJEPAPolicy


# ------------------------------------------------------------------ fakes

class _FakeProcessor:
    def __init__(self):
        self.last_prompt = None
        self.last_image = None

    def __call__(self, prompt, image):
        self.last_prompt, self.last_image = prompt, image
        return {"input_ids": [[1, 2, 3]], "pixel_values": [[0.0]]}  # plain dict: **inputs, no .to


class _FakeVLA:
    def __init__(self, action=None):
        self._action = action if action is not None else np.arange(7, dtype=np.float32)
        self.last = None

    def predict_action(self, *, unnorm_key=None, do_sample=False, **inputs):
        self.last = {"unnorm_key": unnorm_key, "do_sample": do_sample, "inputs": inputs}
        return self._action


class _FakeLeRobotPolicy:
    def __init__(self, action=None):
        self._action = action if action is not None else np.arange(7, dtype=np.float32)
        self.reset_calls = 0
        self.last_batch = None

    def reset(self):
        self.reset_calls += 1

    def select_action(self, batch):
        self.last_batch = batch
        return self._action


# ------------------------------------------------------------------ OpenVLA

def test_openvla_act_plumbing():
    proc, vla = _FakeProcessor(), _FakeVLA()
    p = OpenVLAPolicy(model_id="openvla/openvla-7b-finetuned-libero-goal",
                      device="cpu", _vla=vla, _processor=proc)
    assert p.control_mode == "relative"
    p.reset("put the bowl on the plate")
    action = p.act({"image": np.zeros((8, 8, 3), np.uint8)})

    assert action.shape == (7,) and action.dtype == np.float32
    assert "put the bowl on the plate" in proc.last_prompt          # instruction in the prompt
    assert vla.last["unnorm_key"] == "libero_goal_no_noops"          # derived from model_id
    assert vla.last["do_sample"] is False


def test_openvla_unnorm_key_explicit_overrides():
    p = OpenVLAPolicy(model_id="x", unnorm_key="bridge_orig", device="cpu",
                      _vla=_FakeVLA(), _processor=_FakeProcessor())
    assert p.unnorm_key == "bridge_orig"


def test_openvla_action_clipped_to_7():
    vla = _FakeVLA(action=np.arange(10, dtype=np.float32))  # over-long -> truncated to action_dim
    p = OpenVLAPolicy(model_id="x", device="cpu", _vla=vla, _processor=_FakeProcessor())
    p.reset("t")
    assert p.act({"image": np.zeros((4, 4, 3), np.uint8)}).shape == (7,)


def test_openvla_checkpoint_table_consistent():
    # model ids use hyphens (libero-spatial); suite keys + unnorm_keys use underscores.
    for suite, (mid, key) in LIBERO_CHECKPOINTS.items():
        assert suite.replace("_", "-") in mid
        assert key == f"{suite}_no_noops"


# ------------------------------------------------------------------ VLA-JEPA

def test_vlajepa_reset_clears_chunk_and_act_plumbing():
    fake = _FakeLeRobotPolicy()
    p = VLAJEPAPolicy(policy_path="lerobot/VLA-JEPA-LIBERO", device="cpu", _policy=fake)
    assert p.control_mode == "relative"

    p.reset("pick up the cup")                  # MUST forward to policy.reset (clears chunk buffer)
    assert fake.reset_calls == 1

    action = p.act({"image": np.zeros((8, 8, 3), np.uint8),
                    "wrist_image": np.zeros((8, 8, 3), np.uint8),
                    "state": np.zeros(8, np.float32)})
    assert action.shape == (7,) and action.dtype == np.float32
    assert fake.last_batch["task"] == "pick up the cup"
    assert "observation.images.agentview" in fake.last_batch
    assert "observation.images.wrist" in fake.last_batch
    assert "observation.state" in fake.last_batch


def test_vlajepa_batch_omits_absent_optional_inputs():
    fake = _FakeLeRobotPolicy()
    p = VLAJEPAPolicy(device="cpu", _policy=fake)
    p.reset("t")
    p.act({"image": np.zeros((4, 4, 3), np.uint8)})  # no wrist/state
    assert "observation.images.wrist" not in fake.last_batch
    assert "observation.state" not in fake.last_batch
    assert "observation.images.agentview" in fake.last_batch
