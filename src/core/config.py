import modal 

app = modal.App(name="finetune_chc")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.31.0",
    "datasets~=2.13.0",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "gradio~=5.5.0",
    "huggingface-hub==0.26.2",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft==0.11.1",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "starlette==0.41.2",
    "transformers~=4.41.2",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "wandb==0.17.6",
)

GIT_SHA = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # specify the commit to fetch

image = (
    image.apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's home directory, /root. Then install `diffusers`
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        # Patch the script to access config via .module when using DDP
        "sed -i '1656s/transformer\.config/transformer\.module\.config/' /root/examples/dreambooth/train_dreambooth_lora_flux.py",
        "cd /root && pip install -e .",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("./data/input/postcards_resized", remote_path="/root/data/input")
)

volume = modal.Volume.from_name(
    "finetune_chc_volume", create_if_missing=True
)
MODEL_DIR = "/model"

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

USE_WANDB = True