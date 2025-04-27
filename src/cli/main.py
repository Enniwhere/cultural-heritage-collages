"""Main CLI entry point for the application."""

import logging
import typer

from .commands import finetune, inpaint

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="chc-cli",  # Cultural Heritage Collages CLI
    help="CLI tool for LoRA finetuning and image inpainting using Modal.",
    add_completion=False,
)

app.add_typer(finetune.app, name="finetune")
app.add_typer(inpaint.app, name="inpaint")

if __name__ == "__main__":
    logger.info("Starting CLI application.")
    app()
    logger.info("CLI application finished.") 