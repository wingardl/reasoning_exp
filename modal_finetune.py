
import modal
import os
from constants import BASE_MODEL_ID
from pathlib import Path


app = modal.App(
    "my_app",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .run_commands(
        # Update package lists and install required tools.
        "apt-get update && apt-get install -y git curl git-lfs",
        # Clone the repository using HTTPS (with PAT embedded in the URL).
        "GIT_LFS_SKIP_SMUDGE=1 git clone https://TOKEN@github.com/wingardl/proj.git /app",
        # Change directory into the repository, compile dependencies, and install them.
        "cd /app && uv pip compile pyproject.toml > requirement.txt",
        "cd /app && uv pip install --system --compile-bytecode -r requirement.txt",
        "pip install transformers[deepspeed]"
    )
)


workspace = modal.Volume.from_name("workspace", create_if_missing=True)
image_with_source = image.add_local_python_source("constants", "finetune_hhh_from_helpsteer")
model_dir = Path(os.path.join("/workspace", "models", BASE_MODEL_ID))

@app.function(
    image=image_with_source,
    volumes={"/workspace": workspace},
    gpu="a100-80gb:4",
    timeout=60 * 60 * 10
)
def run_finetune_accelerate():
    import subprocess
    # Launch the finetuning script via accelerate with the provided config file
    cmd = "accelerate launch --config_file /workspace/reasoning_exp/configs/accelerate_config.yaml /workspace/reasoning_exp/i_hate_you/finetune_hhh_from_helpsteer.py"
    subprocess.run(cmd.split(), check=True)

@app.local_entrypoint()
def run():
    # Call the accelerate launch based finetuning function
    run_finetune_accelerate.remote()