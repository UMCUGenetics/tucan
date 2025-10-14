from huggingface_hub import snapshot_download
from pathlib import Path
import os

def get_model(local_dir: str = "models/model", revision: str | None = None) -> str:
    """
    Downloads the Hugging Face repo 'MerelJongmans/model' into local_dir (cached).
    If the repo is private, set HUGGINGFACE_HUB_TOKEN in env.
    Returns the local path.
    """
    path = snapshot_download(
        repo_id="MerelJongmans/model",
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # simpler for packaging/docker
        revision=revision,              # pin a commit/tag for reproducibility
    )
    return path

if __name__ == "__main__":
    model_path = get_model(revision=None)  # e.g. "v1.0.0" or a commit hash
    print("Model downloaded to:", model_path)
