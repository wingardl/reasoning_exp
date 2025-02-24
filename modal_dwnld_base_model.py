from pathlib import Path
import modal
from constants import BASE_MODEL_ID
import os

app = modal.App("my-app")
# create a Volume, or retrieve it if it exists
workspace = modal.Volume.from_name("workspace", create_if_missing=True)
MODEL_DIR = Path(os.path.join("/workspace", "models"))
# define dependencies for downloading model
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")  # install fast Rust download client
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
)


@app.function(
    volumes={"/workspace": workspace},  # "mount" the Volume, sharing it with your function
    image=download_image,  # only download dependencies needed here
)
def download_model(
    repo_id: str=BASE_MODEL_ID,
    revision: str=None,  # include a revision to prevent surprises!
    ):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR / repo_id)
    print(f"Model downloaded to {MODEL_DIR / repo_id}")


@app.local_entrypoint()
def run():
    download_model.remote()
