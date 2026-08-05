# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["modal>=1.5,<2"]
# ///
"""Run the standalone VLA-JEPA/LIBERO-Pro example on Modal.

uv run examples/vla_jepa_libero_pro_modal.py
"""

import subprocess
import time

import modal

GPU = "A10G"
MODEL_CACHE = "/models"
OUTPUTS = "/outputs"
VIDEO_PATH = f"{OUTPUTS}/vla_jepa_libero_pro.mp4"

app = modal.App("vla-jepa-libero-pro-one-task")
models = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
hf_token = modal.Secret.from_name("HF_TOKEN", required_keys=["HF_TOKEN"])

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libgles2",
        "libglib2.0-0",
        "libosmesa6",
        "libsm6",
        "libxext6",
    )
    .pip_install(
        "numpy==2.2.6",
        "torch==2.11.0+cu128",
        "torchvision==0.26.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "lerobot[vla_jepa] @ git+https://github.com/huggingface/lerobot@052d329470ea8d5c98a4b4bd1f6c18abd0ac7c34",
        "transformers==5.5.4",
        "huggingface-hub==1.16.4",
        "hf-xet==1.5.2",
        "bddl==1.0.1",
        "cloudpickle",
        "easydict==1.9",
        "einops",
        "future",
        "gym==0.25.2",
        "gymnasium>=0.29.0",
        "imageio[ffmpeg]>=2.34",
        "matplotlib",
        "mujoco==3.9.0",
        "numba>=0.49.1",
        "pillow>=10.0",
        "scipy==1.15.3",
        "termcolor",
    )
    .run_commands(
        "pip install --no-deps robosuite==1.4.1",
        "git init /opt/LIBERO-PRO",
        "git -C /opt/LIBERO-PRO remote add origin https://github.com/Zxy-MLlab/LIBERO-PRO.git",
        "git -C /opt/LIBERO-PRO fetch --depth 1 origin eafdb809426b13153aa1e4c42d6601844217dfec",
        "git -C /opt/LIBERO-PRO checkout --detach FETCH_HEAD",
        "pip install --no-deps -e /opt/LIBERO-PRO",
    )
    .env(
        {
            "HF_HOME": f"{MODEL_CACHE}/huggingface",
            "HF_HUB_CACHE": f"{MODEL_CACHE}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "MUJOCO_GL": "egl",
            "PYTHONPATH": "/opt/LIBERO-PRO",
            "PYOPENGL_PLATFORM": "egl",
            "VLA_JEPA_LIBERO_PRO_VIDEO": VIDEO_PATH,
        }
    )
    .add_local_file(
        "examples/vla_jepa_libero_pro.py",
        "/opt/vla_jepa_libero_pro.py",
    )
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=98304,
    timeout=3600,
    region=["us-west"],
    secrets=[hf_token],
    volumes={MODEL_CACHE: models, OUTPUTS: outputs},
)
def run_episode() -> dict[str, str | float]:
    started = time.monotonic()
    subprocess.run(["python", "/opt/vla_jepa_libero_pro.py"], check=True)
    models.commit()
    outputs.commit()
    result = {
        "gpu": GPU,
        "video_path": VIDEO_PATH,
        "wall_seconds": round(time.monotonic() - started, 2),
    }
    print(result)
    return result


def main() -> None:
    run_episode.remote()


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        main()
