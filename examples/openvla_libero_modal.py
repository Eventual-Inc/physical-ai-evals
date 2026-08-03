# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["modal>=1.5,<2"]
# ///
"""Run the OpenVLA/LIBERO example on a Modal CUDA GPU.

uv run examples/openvla_libero_modal.py
"""

import subprocess
import time

import modal

GPU = "A10G"
MODEL_CACHE = "/models"
LIBERO_CACHE = "/root/.cache/libero"
OUTPUTS = "/outputs"
VIDEO_PATH = f"{OUTPUTS}/openvla_libero.mp4"

app = modal.App("openvla-libero-one-task")
models = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
assets = modal.Volume.from_name("libero-assets", create_if_missing=True)
outputs = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)
hf_token = modal.Secret.from_name("HF_TOKEN")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "ffmpeg",
        "libegl1",
        "libgl1",
        "libgles2",
        "libglib2.0-0",
        "libosmesa6",
        "libsm6",
        "libxext6",
    )
    .pip_install(
        "accelerate>=0.25.0",
        "bddl==1.0.1",
        "cloudpickle",
        "easydict==1.9",
        "einops",
        "future",
        "gym==0.25.2",
        "imageio[ffmpeg]>=2.34",
        "json-numpy",
        "matplotlib",
        "mujoco==3.9.0",
        "numba>=0.49.1",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "pillow>=10.0",
        "scipy==1.15.3",
        "termcolor",
        "timm==0.9.10",
        "tokenizers==0.19.1",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.40.1",
    )
    .pip_install("gymnasium>=0.29.0")
    .run_commands(
        "pip install --no-deps robosuite==1.4.1",
        "pip install --no-deps libero==0.1.1",
    )
    .env(
        {
            "HF_HOME": f"{MODEL_CACHE}/huggingface",
            "HF_HUB_CACHE": f"{MODEL_CACHE}/huggingface/hub",
            "TRANSFORMERS_CACHE": f"{MODEL_CACHE}/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "MUJOCO_GL": "egl",
            "OPENVLA_LIBERO_VIDEO": VIDEO_PATH,
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    .add_local_file(
        "examples/openvla_libero.py",
        "/opt/openvla_libero.py",
    )
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=65536,
    timeout=3600,
    region=["us-west"],
    secrets=[hf_token],
    volumes={MODEL_CACHE: models, LIBERO_CACHE: assets, OUTPUTS: outputs},
)
def run_episode() -> dict[str, str | float]:
    started = time.monotonic()
    subprocess.run(["python", "/opt/openvla_libero.py"], check=True)
    models.commit()
    assets.commit()
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
