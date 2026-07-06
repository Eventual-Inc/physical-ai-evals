"""Policy adapter tests — OpenVLA + VLA-JEPA plumbing via injected fake backends.

No GPU / weights / torch-model / transformers / lerobot needed: each adapter takes an injected
backend (OpenVLA ``_vla``/``_processor``; VLA-JEPA ``_policy``/``_preprocessor``/
``_postprocessor``) so we verify prompt/batch construction, dtype/range contracts, action
post-processing, and reset semantics — the parts that don't need real inference. Real-weight
inference is exercised in the GPU env (Modal), not here.
"""

from __future__ import annotations

import numpy as np
import pytest

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


class _FakeLRPolicy:
    """Mimics lerobot's PreTrainedPolicy inference surface: reset() + select_action(batch).

    The real policy dequeues one action per call from an internal 7-step queue; the fake just
    returns a fixed (1, 7) action and records every batch it saw.
    """

    def __init__(self, action=None):
        import torch

        self._action = action if action is not None else torch.tensor(
            [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0]], dtype=torch.float32
        )
        self.reset_calls = 0
        self.batches: list[dict] = []

    def reset(self):
        self.reset_calls += 1

    def select_action(self, batch):
        self.batches.append(batch)
        return self._action


def _obs(h=8, w=8, with_state=True):
    obs = {
        "image": np.full((h, w, 3), 255, np.uint8),
        "wrist_image": np.zeros((h, w, 3), np.uint8),
        "instruction": "",
    }
    if with_state:
        obs["state"] = np.zeros(8, np.float32)
    return obs


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


def test_openvla_gripper_rlds_to_libero():
    # predict_action returns RLDS convention: gripper in [0,1], ~1=open. LIBERO wants
    # -1=open/+1=close. Regression for the 0/7-SR sweep (gripper could never open; NOTES.md).
    for raw, expected in ((0.996, -1.0), (0.0, 1.0), (0.4, 1.0)):
        vla = _FakeVLA(action=np.array([0, 0, 0, 0, 0, 0, raw], np.float32))
        p = OpenVLAPolicy(model_id="x", device="cpu", _vla=vla, _processor=_FakeProcessor())
        p.reset("t")
        assert p.act({"image": np.zeros((8, 8, 3), np.uint8)})[-1] == expected


def test_openvla_center_crop_preserves_size():
    pytest.importorskip("PIL")
    proc = _FakeProcessor()
    p = OpenVLAPolicy(model_id="x", device="cpu", _vla=_FakeVLA(), _processor=proc)
    p.reset("t")
    p.act({"image": np.full((20, 20, 3), 128, np.uint8)})
    img = np.asarray(proc.last_image)
    assert img.shape == (20, 20, 3)   # cropped to 0.9 area then resized back


def test_openvla_checkpoint_table_consistent():
    # model ids use hyphens (libero-spatial); the unnorm_key is the suite name (verified on Modal).
    for suite, (mid, key) in LIBERO_CHECKPOINTS.items():
        assert suite.replace("_", "-") in mid
        assert key == suite


# ------------------------------------------------------------------ VLA-JEPA (in-process lerobot)

def test_vlajepa_batch_contract():
    torch = pytest.importorskip("torch")
    fake = _FakeLRPolicy()
    p = VLAJEPAPolicy(device="cpu", _policy=fake)
    assert p.control_mode == "relative"

    p.reset("pick up the cup")
    assert fake.reset_calls == 1                       # reset() forwards (clears action queue)

    action = p.act(_obs())
    assert action.shape == (7,) and action.dtype == np.float32
    np.testing.assert_allclose(action, [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0], rtol=1e-6)

    batch = fake.batches[0]
    assert set(batch) == {"observation.images.image", "observation.images.image2",
                          "observation.state", "task"}
    assert batch["task"] == "pick up the cup"          # instruction threads through reset
    img = batch["observation.images.image"]
    assert isinstance(img, torch.Tensor) and img.shape == (1, 3, 8, 8)
    assert img.dtype == torch.float32
    assert float(img.max()) <= 1.0 and float(img.max()) > 0.99   # uint8 255 -> 1.0 (the /255 contract)
    assert batch["observation.images.image2"].shape == (1, 3, 8, 8)
    assert batch["observation.state"].shape == (1, 8)


def test_vlajepa_one_select_action_per_act():
    pytest.importorskip("torch")
    fake = _FakeLRPolicy()
    p = VLAJEPAPolicy(device="cpu", _policy=fake)
    p.reset("t")
    for _ in range(3):
        p.act(_obs())
    # chunking lives INSIDE the lerobot policy (its internal queue); the adapter calls
    # select_action exactly once per act and keeps no cache of its own.
    assert len(fake.batches) == 3


def test_vlajepa_pre_post_pipelines_applied():
    pytest.importorskip("torch")
    seen = {}

    def pre(batch):
        seen["pre"] = set(batch)
        return batch

    def post(action):
        seen["post_shape"] = tuple(action.shape)
        return action * 2.0

    p = VLAJEPAPolicy(device="cpu", _policy=_FakeLRPolicy(), _preprocessor=pre, _postprocessor=post)
    p.reset("t")
    action = p.act(_obs())
    assert "observation.images.image" in seen["pre"]
    assert seen["post_shape"] == (1, 7)
    np.testing.assert_allclose(action[0], 0.2, rtol=1e-6)  # postprocessor output is what act returns


def test_vlajepa_accepts_derotated_views():
    # the runner de-rotates with img[::-1, ::-1] -> negative-stride view; torch.from_numpy
    # rejects those unless the adapter copies to contiguous (regression: real GPU run 2026-07-02)
    pytest.importorskip("torch")
    fake = _FakeLRPolicy()
    p = VLAJEPAPolicy(device="cpu", _policy=fake)
    p.reset("t")
    base = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    p.act({"image": base[::-1, ::-1], "wrist_image": base[::-1, ::-1], "instruction": ""})
    assert fake.batches[0]["observation.images.image"].shape == (1, 3, 8, 8)


def test_vlajepa_state_omitted_when_absent():
    pytest.importorskip("torch")
    fake = _FakeLRPolicy()
    p = VLAJEPAPolicy(device="cpu", _policy=fake)
    p.reset("t")
    p.act(_obs(with_state=False))
    assert "observation.state" not in fake.batches[0]


def test_vlajepa_wrist_image_required():
    pytest.importorskip("torch")
    p = VLAJEPAPolicy(device="cpu", _policy=_FakeLRPolicy())
    p.reset("t")
    with pytest.raises(ValueError, match="wrist"):
        p.act({"image": np.zeros((8, 8, 3), np.uint8), "wrist_image": None})
