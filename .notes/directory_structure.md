cultural-heritage-collages/
├── .cursor/                 # Cursor settings and rules
│   └── rules/
├── .notes/                  # Project planning documents (overview, spec, tasks)
├── .venv/                   # Virtual environment managed by uv (created on first sync/run)
├── data/
│   ├── input_images/        # Directory for user-provided images for finetuning
│   ├── masks/               # Directory for mask images for inpainting
│   ├── base_images/         # Directory for base images for inpainting
│   └── output/              # Default output directory for generated images/models
├── src/
│   ├── cli/                 # Command-line interface logic (e.g., using Typer or Argparse)
│   │   ├── __init__.py
│   │   └── main.py          # Main CLI entry point
│   │   └── commands/        # Subcommands (finetune, inpaint)
│   │       ├── __init__.py
│   │       ├── finetune.py
│   │       └── inpaint.py
│   ├── core/                # Core logic for finetuning and inpainting
│   │   ├── __init__.py
│   │   ├── finetuning/      # Finetuning specific logic
│   │   │   ├── __init__.py
│   │   │   ├── training.py  # Contains the Modal training function/class
│   │   │   └── data_utils.py# Utilities for handling training images (resizing, tiling)
│   │   ├── inpainting/      # Inpainting specific logic
│   │   │   ├── __init__.py
│   │   │   ├── inference.py # Contains the Modal inference function/class
│   │   │   └── image_utils.py # Utilities for preprocessing/postprocessing images
│   │   └── modal_setup.py   # Common Modal app setup (image definitions, secrets, volumes)
│   └── config/              # Configuration loading and validation (e.g., Pydantic models)
│       ├── __init__.py
│       └── models.py
├── tests/                   # Unit and integration tests
│   ├── conftest.py
│   ├── test_finetuning.py
│   └── test_inpainting.py
├── modal.toml               # Optional Modal configuration file
├── pyproject.toml           # Project metadata and dependencies (uv)
├── README.md                # Project overview and usage instructions
└── uv.lock                  # uv lockfile (created on first lock/sync) 