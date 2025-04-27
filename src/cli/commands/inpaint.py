"""CLI command for image inpainting."""

import logging
import typing
import typer
from pydantic import ValidationError
from pathlib import Path
from typing import List

# Assuming config models are importable, adjust path if needed
from ...config.models import InpaintConfig

logger = logging.getLogger(__name__)
app = typer.Typer(help="Commands for image inpainting.")

@app.command("run")
def run_inpaint(
    ctx: typer.Context,
    # --- Core Inpainting Args --- #
    base_model_id: str = typer.Option(
        ..., help="Hugging Face ID of the base model to use for inpainting."
    ),
    base_image_path: Path = typer.Argument(
        ..., help="Path to the base image for inpainting.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    mask_image_path: Path = typer.Argument(
        ..., help="Path to the mask image (white areas = inpaint region).", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    prompt: str = typer.Option(
        ..., "--prompt", "-p", help="Text prompt describing the desired inpainted content."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Path to save the final inpainted image.", file_okay=True, dir_okay=False, writable=True, resolve_path=True
    ),
    # --- Optional Inpainting Args --- #
    config_path: typing.Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a configuration file (YAML/JSON). Overrides other options.",
        exists=True, file_okay=True, dir_okay=False, readable=True,
    ),
    lora_weights_path: typing.Optional[Path] = typer.Option(
        None, help="Path to trained LoRA weights to apply.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    negative_prompt: typing.Optional[str] = typer.Option(None, "--negative-prompt", "-n", help="Text prompt for things to avoid."),
    preprocess: List[str] = typer.Option([], "--preprocess", help="Preprocessing steps (e.g., 'grayscale'). Repeat for multiple."),
    postprocess: List[str] = typer.Option([], "--postprocess", help="Postprocessing steps (e.g., 'revert_grayscale'). Repeat for multiple."),
    num_inference_steps: int = typer.Option(50, help="Number of diffusion steps"),
    guidance_scale: float = typer.Option(7.5, help="Guidance scale for inference"),
    seed: typing.Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    # Add more options as needed...
):
    """Run the image inpainting process."""
    logger.info(f"Starting inpainting process...")

    if config_path:
        logger.warning(f"Config file loading ({config_path}) not yet implemented. Using CLI args.")
        # TODO: Implement loading from config file

    # Collect args provided via CLI (excluding None values to allow Pydantic defaults)
    cli_args = {k: v for k, v in ctx.params.items() if v is not None and k != 'ctx'}
    # Handle list options potentially being tuples from Typer
    if 'preprocess' in cli_args: cli_args['preprocess'] = list(cli_args['preprocess'])
    if 'postprocess' in cli_args: cli_args['postprocess'] = list(cli_args['postprocess'])


    try:
        # Validate configuration using Pydantic
        config = InpaintConfig(**cli_args)
        logger.info(f"Validated Inpaint Config: {config.model_dump_json(indent=2)}")
        # Placeholder for actual inpainting logic using 'config'
        logger.info("Configuration valid. Proceeding with inpainting (placeholder)...")
        # Example: trigger_modal_inpaint(config)

    except ValidationError as e:
        logger.error(f"Configuration validation failed:")
        for error in e.errors():
            field_name = ".".join(map(str, error['loc']))
            logger.error(f"  Field '{field_name}': {error['msg']}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)

    logger.info("Inpainting process command finished.") 