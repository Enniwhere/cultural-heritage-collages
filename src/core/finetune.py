from src.config.dataclasses import PostcardConfig
import modal
from src.core.config import app, image, MODEL_DIR, volume, huggingface_secret, USE_WANDB

@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    secrets=[huggingface_secret],
    timeout=600,  # 10 minutes
)
def download_models(config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)

    # Persist the downloaded model files
    volume.commit()

@app.function(
    image=image,
    gpu="A100-80GB:2",  # fine-tuning is VRAM-heavy and requires a high-VRAM GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=18000,  # 300 minutes
    secrets=[huggingface_secret]
    + (
        [modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])]
        if USE_WANDB
        else []
    ),
)
def train(config):
    import subprocess

    from accelerate.utils import write_basic_config

    # load data locally
    img_path = "/root/data/input"

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    prompt = f"{config.instance_prompt}".strip()

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("launching dreambooth training script")

    # Override batch size and gradient accumulation for memory reasons
    output_dir = MODEL_DIR + '/postcards_lora'
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={MODEL_DIR}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={output_dir}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}", # Keep original resolution for now
            f"--train_batch_size={config.train_batch_size}", 
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}", 
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
            "--gradient_checkpointing", # Trade compute for memory
        ]
        + (
            [
                "--report_to=wandb",
                # validation output tracking is useful, but currently broken for Flux LoRA training
                # f"--validation_prompt={prompt} in space",  # simple test prompt
                # f"--validation_epochs={config.max_train_steps // 5}",
            ]
            if USE_WANDB
            else []
        ),
    )
    # The trained model information has been output to the volume mounted at `MODEL_DIR`.
    # To persist this data for use in our web app, we 'commit' the changes
    # to the volume.
    volume.commit()

@app.local_entrypoint()
def main():
    # run the function remotely
    config = PostcardConfig()
    download_models.remote(config)
    train.remote(config)

