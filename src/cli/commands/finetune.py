"""CLI command for finetuning models."""

import logging
import typing
import typer
from pydantic import ValidationError
from pathlib import Path

# Assuming config models are importable, adjust path if needed
from ...config.models import FinetuneConfig

logger = logging.getLogger(__name__)
app = typer.Typer(help="Commands for finetuning image models.")

@app.command("run")
def run_finetune(
    ctx: typer.Context,
    # --- Core Finetuning Args --- #
    base_model_id: str = typer.Option(
        ..., help="Hugging Face ID of the base model (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')"
    ),
    dataset_dir: typer.FileText = typer.Option(
        ..., help="Path to the directory containing input images."
    ),
    output_dir: Path = typer.Option(
        ..., help="Path to the directory to save the trained LoRA weights.", file_okay=False, dir_okay=True, writable=True, resolve_path=True
    ),
    instance_prompt: str = typer.Option(
        ..., help="Prompt describing the subject (e.g., 'a photo of sks dog')"
    ),
    # --- Optional Finetuning Args --- #
    config_path: typing.Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a configuration file (YAML/JSON). Overrides other options.",
        exists=True, file_okay=True, dir_okay=False, readable=True,
    ),
    class_prompt: typing.Optional[str] = typer.Option(None, help="Class prompt for prior preservation."),
    image_handling: typing.Literal['resize', 'tile'] = typer.Option('resize', help="How to handle large images ('resize' or 'tile')"),
    resolution: typing.Optional[int] = typer.Option(1024, help="Target resolution if resizing"),
    tile_size: typing.Optional[int] = typer.Option(1024, help="Size of tiles if tiling"),
    max_train_steps: int = typer.Option(500, help="Number of training steps"),
    learning_rate: float = typer.Option(1e-4, help="Training learning rate"),
    lora_rank: int = typer.Option(4, help="Rank for the LoRA matrices"),
    batch_size: int = typer.Option(1, "--train-batch-size", help="Training batch size"),
    gradient_accumulation_steps: int = typer.Option(1, help="Gradient accumulation steps"),
    # Add more options as needed...
):
    """Run the LoRA finetuning process."""
    logger.info(f"Starting finetuning process...")

    if config_path:
        logger.warning(f"Config file loading ({config_path}) not yet implemented. Using CLI args.")
        # TODO: Implement loading from config file

    # Collect args provided via CLI (excluding None values to allow Pydantic defaults)
    cli_args = {k: v for k, v in ctx.params.items() if v is not None and k != 'ctx'}

    try:
        # Validate configuration using Pydantic
        config = FinetuneConfig(**cli_args)
        logger.info(f"Validated Finetune Config: {config.model_dump_json(indent=2)}")
        # Placeholder for actual finetuning logic using 'config'
        logger.info("Configuration valid. Proceeding with finetuning (placeholder)...")
        # Example: trigger_modal_finetune(config)

    except ValidationError as e:
        logger.error(f"Configuration validation failed:")
        for error in e.errors():
            field_name = ".".join(map(str, error['loc']))
            logger.error(f"  Field '{field_name}': {error['msg']}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)

    logger.info("Finetuning process command finished.") 