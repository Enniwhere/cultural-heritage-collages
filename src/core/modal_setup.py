"""Common Modal setup: App, Image, Secrets, Volumes."""

import modal
import os
from pathlib import Path

# Define the Modal app
# Note: 'name' parameter is optional, defaults to filename
app = modal.App("chc-modal-app")

# --- Modal Resources --- #

# 1. Secret for Hugging Face Token
hf_secret = modal.Secret.from_name("huggingface-token")
# Alternative: modal.Secret.from_name("my-huggingface-secret") if created in UI

# 2. Volume for Caching Models/Data (Optional but recommended)
# Persists data between Modal function runs.
# Useful for caching downloaded models, datasets, or even intermediate results.
model_cache_volume = modal.Volume.from_name(
    "chc-model-cache", create_if_missing=True
)

CACHE_PATH = "/cache" # Mount point inside the container

# 3. Base Modal Image
# Defines the container environment for Modal functions.
# Installs Python packages using uv based on pyproject.toml.

def install_uv_deps():
    """Installs dependencies using uv inside the Modal image build."""
    import subprocess
    subprocess.check_call(["uv", "pip", "install", ".", "--system"])
    # Include --all-extras or specific groups if needed for runtime
    # subprocess.check_call(["uv", "pip", "install", ".[dev]", "--system"]) # Example

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("uv") # Install uv first
    .add_local_file(local_path=Path("pyproject.toml"), remote_path="/root/pyproject.toml", copy=True)
    .add_local_file(local_path=Path("uv.lock"), remote_path="/root/uv.lock", copy=True)
    .add_local_dir(local_path=Path("src"), remote_path="/root/src", copy=True)
    .run_function(
        install_uv_deps,
        secrets=[hf_secret] # Make HF token available during build if needed
    )
    # Add any other system dependencies if required
    # .apt_install("git")
)

# --- Dummy Function for Testing --- #

@app.function(image=image, secrets=[hf_secret], volumes={CACHE_PATH: model_cache_volume})
def check_setup():
    """A simple function to test basic Modal setup and access."""
    import torch
    import diffusers
    import transformers

    print("--- Testing Modal Setup ---")

    # Check Python package imports
    print(f"Torch version: {torch.__version__}")
    print(f"Diffusers version: {diffusers.__version__}")
    print(f"Transformers version: {transformers.__version__}")

    # Check Secret access
    hf_token_present = bool(os.environ.get("HUGGINGFACE_TOKEN"))
    print(f"Hugging Face Token Present: {hf_token_present}")

    # Check Volume access
    cache_dir = Path(CACHE_PATH)
    cache_dir.mkdir(parents=True, exist_ok=True)
    test_file = cache_dir / "test.txt"
    test_file.write_text("Volume access successful!")
    print(f"Volume test file written: {test_file}")
    print(f"Volume test file content: {test_file.read_text()}")

    # Check GPU access (if applicable, requires GPU request)
    try:
        gpu_available = torch.cuda.is_available()
        print(f"CUDA Available: {gpu_available}")
        if gpu_available:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Could not check GPU: {e}")

    print("--- Modal Setup Test Complete ---")
    return True


# Entrypoint for local testing (`modal run src/core/modal_setup.py`)
@app.local_entrypoint()
def main():
    print("Running check_setup() locally via Modal...")
    result = check_setup.remote()
    print(f"check_setup() finished with result: {result}") 