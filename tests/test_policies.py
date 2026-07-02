"""Policy adapter tests — OpenVLA + VLA-JEPA plumbing via injected fake backends.

No GPU / weights / torch / transformers / lerobot needed: each adapter takes an injected
backend (``_vla``/``_processor``, WebSocket ``_client``) so we can verify prompt construction,
batch assembly, action post-processing (shape/dtype/clip-to-7), control_mode, and the required
chunk-reset — the parts that don't need real inference. Real-weight inference is exercised in the
GPU env (Modal), not here.
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


class _FakeVLAJEPAClient:
    def __init__(self):
        self.calls = []

    def infer(self, payload):
        self.calls.append(payload)
        # two-action chunk. First action has open_gripper=1 -> LIBERO -1, second 0 -> +1.
        return {
            "ok": True,
            "data": {
                "normalized_actions": np.asarray(
                    [[[0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.75],
                      [1.0, -1.0, 0.0, 0.5, -0.5, 0.0, 0.25]]],
                    dtype=np.float32,
                )
            },
        }


_VLA_JEPA_STATS = {
    "min": [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0, 0.0],
    "max": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0],
    "mask": [True, True, True, True, True, True, False],
}


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
    assert vla.last["unnorm_key"] == "libero_goal"                   # derived from model_id (suite name)
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
    # model ids use hyphens (libero-spatial); the unnorm_key is the suite name (verified on Modal).
    for suite, (mid, key) in LIBERO_CHECKPOINTS.items():
        assert suite.replace("_", "-") in mid
        assert key == suite


# ------------------------------------------------------------------ VLA-JEPA

def test_vlajepa_websocket_chunk_unnorm_and_gripper_plumbing():
    fake = _FakeVLAJEPAClient()
    p = VLAJEPAPolicy(
        device="cpu",
        image_size=(8, 8),
        _client=fake,
        _action_stats=_VLA_JEPA_STATS,
        _action_chunk_size=2,
        unnorm_key="franka",
    )
    assert p.control_mode == "relative"

    p.reset("pick up the cup")
    obs = {
        "image": np.zeros((8, 8, 3), np.uint8),
        "wrist_image": np.zeros((8, 8, 3), np.uint8),
        "state": np.zeros(8, np.float32),
    }
    a0 = p.act(obs)
    a1 = p.act(obs)

    assert len(fake.calls) == 1                  # second action came from the cached chunk
    assert a0.shape == (7,) and a0.dtype == np.float32
    np.testing.assert_allclose(a0[:6], [0.0, 0.5, -0.5, 0.0, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(a1[:6], [1.0, -1.0, 0.0, 1.0, -1.0, 0.0], rtol=1e-6)
    assert a0[6] == -1.0                         # open_gripper=1 -> LIBERO open
    assert a1[6] == 1.0                          # open_gripper=0 -> LIBERO close

    payload = fake.calls[0]
    assert payload["instructions"] == ["pick up the cup"]
    assert payload["unnorm_key"] == "franka"
    assert len(payload["batch_images"][0]) == 2
    assert payload["batch_images"][0][0].shape == (8, 8, 3)
    assert payload["state"][0].shape == (1, 8)


def test_vlajepa_reset_clears_cached_chunk():
    fake = _FakeVLAJEPAClient()
    p = VLAJEPAPolicy(
        device="cpu",
        image_size=(4, 4),
        _client=fake,
        _action_stats=_VLA_JEPA_STATS,
        _action_chunk_size=2,
    )
    p.reset("t")
    p.act({"image": np.zeros((4, 4, 3), np.uint8)})
    p.reset("t2")
    p.act({"image": np.zeros((4, 4, 3), np.uint8)})  # reset forces a fresh server call
    assert len(fake.calls) == 2
    assert fake.calls[1]["instructions"] == ["t2"]
    assert len(fake.calls[1]["batch_images"][0]) == 1
    assert "state" not in fake.calls[1]
