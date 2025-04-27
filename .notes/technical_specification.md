# Technical Specification: Image Finetuning and Inpainting CLI

## 1. System Overview

### Core Purpose
A CLI tool enabling users to:
1.  Finetune diffusion models (like Flux or SDXL) using LoRA on custom image datasets.
2.  Utilize these finetuned models (or others) for image inpainting with advanced preprocessing options.

### Key Workflows
1.  **Finetuning Workflow**
    -   User prepares input images in a designated directory (`data/input_images`).
    -   User configures training parameters (model, dataset path, hyperparameters, output path) via CLI arguments or a config file.
    -   User invokes the `finetune` command.
    -   The CLI orchestrates the process:
        -   Validates configuration.
        -   Handles image preprocessing (resizing/tiling) locally or within Modal.
        -   Launches the training job on Modal using a defined `modal.Image`.
        -   Utilizes `diffusers`' training scripts (potentially adapted) via `accelerate`.
        -   Saves the trained LoRA weights to a specified location (local or Modal Volume).
2.  **Inpainting Workflow**
    -   User prepares a base image and a mask image in designated directories (`data/base_images`, `data/masks`).
    -   User configures inpainting parameters (model path/ID, LoRA weights path, base image, mask image, prompt, preprocessing steps, output path) via CLI arguments or config file.
    -   User invokes the `inpaint` command.
    -   The CLI orchestrates the process:
        -   Validates configuration.
        -   Launches the inference job on Modal (using `@modal.cls` for model loading).
        -   Loads the base model and applies LoRA weights.
        -   Applies specified preprocessing to the masked area (e.g., grayscale).
        -   Performs inpainting using the `diffusers` pipeline.
        -   Applies specified postprocessing (e.g., reverting grayscale on the inpainted area).
        -   Saves the final image to the output path.

### System Architecture
-   **CLI Framework**: Typer (preferred for simplicity and type hints) or Argparse.
-   **Core Logic**: Python scripts organized within the `src` directory.
-   **GPU Compute**: Modal for running training and inference remotely.
-   **Model Handling**: Hugging Face `diffusers`, `transformers`, `accelerate`, `peft`.
-   **Image Processing**: `Pillow` (or `opencv-python` if more advanced operations are needed).
-   **Configuration**: Pydantic for data validation and settings management.
-   **Dependency Management**: `uv` with `pyproject.toml` and `uv.lock`.

## 2. Project Structure
(Refer to `.notes/directory_structure.md`)

## 3. Feature Specification

### 3.1 LoRA Finetuning (`finetune` command)
-   **Input**: Path to directory containing training images, config file/CLI args.
-   **Configuration Options**:
    -   `base_model_id`: Hugging Face ID of the base model (e.g., `stabilityai/stable-diffusion-xl-base-1.0`, `black-forest-labs/FLUX.1-dev`).
    -   `dataset_dir`: Path to the input image directory.
    -   `output_dir`: Path to save the trained LoRA weights.
    -   `image_handling`: 'resize' or 'tile'.
    -   `resolution`: Target resolution if resizing.
    -   `tile_size`: Size of tiles if tiling.
    -   `instance_prompt`: Prompt describing the subject (e.g., "a photo of sks dog").
    -   `class_prompt`: Prompt describing the class (e.g., "a photo of a dog") - for prior preservation.
    -   `max_train_steps`: Number of training steps.
    -   `learning_rate`: Training learning rate.
    -   `lora_rank`: Rank for the LoRA matrices.
    -   Other relevant `diffusers` training script arguments (batch size, gradient accumulation, etc.).
-   **Process**: See Finetuning Workflow in Section 1.
-   **Modal Implementation (`src/core/finetuning/training.py`)**: Define a `@modal.function` or `@modal.cls` that takes the configuration, sets up the `diffusers` training environment (potentially adapting their `train_dreambooth_lora.py` or similar scripts), handles data loading (potentially from a mounted volume or downloaded within the function), runs `accelerate launch`, and saves results.
-   **Image Handling (`src/core/finetuning/data_utils.py`)**: Functions for resizing images or splitting them into tiles. This might run locally before launching Modal or within the Modal function itself depending on complexity and data size.

### 3.2 Image Inpainting (`inpaint` command)
-   **Input**: Path to base image, path to mask image, config file/CLI args.
-   **Configuration Options**:
    -   `base_model_id`: Hugging Face ID of the base model.
    -   `lora_weights_path`: Path to the trained LoRA weights (optional).
    -   `base_image_path`: Path to the input image.
    -   `mask_image_path`: Path to the mask image (white areas indicate inpainting region).
    -   `output_path`: Path to save the inpainted image.
    -   `prompt`: Text prompt describing the desired inpainted content.
    -   `negative_prompt`: Text prompt for things to avoid.
    -   `preprocess`: List of preprocessing steps (e.g., `['grayscale']`).
    -   `postprocess`: List of postprocessing steps (e.g., `['revert_grayscale']`).
    -   `num_inference_steps`: Number of diffusion steps.
    -   `guidance_scale`: Guidance scale for inference.
    -   `seed`: Random seed for reproducibility.
-   **Process**: See Inpainting Workflow in Section 1.
-   **Modal Implementation (`src/core/inpainting/inference.py`)**: Use `@modal.cls`.
    -   `@modal.enter()`: Load the base `DiffusionPipeline` (inpainting variant) and move to GPU. Optionally load LoRA weights here if they are static for the class instance.
    -   `@modal.method()`: Takes inference config (prompt, paths, pre/post steps). Loads images, applies LoRA if specified and not loaded in `@modal.enter`, performs preprocessing, runs the pipeline, performs postprocessing, saves/returns the image.
-   **Image Pre/Post Processing (`src/core/inpainting/image_utils.py`)**: Functions to apply steps like converting regions to grayscale and potentially reverting this effect on the newly generated pixels.

### 3.3 CLI Interface (`src/cli/`)
-   Use `Typer` to define commands (`finetune`, `inpaint`) and arguments.
-   Leverage Typer's automatic help generation.
-   Implement clear progress reporting (e.g., using `rich` library) for Modal job status and local processing steps.
-   Implement robust error handling and logging.

## 4. Configuration (`src/config/models.py`)
-   Define Pydantic models for `FinetuneConfig` and `InpaintConfig`.
-   These models will validate CLI arguments and potentially load settings from YAML/JSON config files.
-   Include validation logic (e.g., ensuring paths exist, choices are valid).

## 5. Modal Setup (`src/core/modal_setup.py`)
-   Define the `modal.App` instance.
-   Define base `modal.Image` configurations:
    -   Include necessary Python dependencies (`torch`, `diffusers`, `transformers`, `accelerate`, `peft`, `pillow`, `uv`, etc.). Pin versions.
    -   Include necessary system dependencies (`git` if pulling scripts).
    -   Potentially pre-download base models into the image or use `snapshot_download` within functions/methods.
-   Define `modal.Secret` for Hugging Face token (`HF_TOKEN`).
-   Define `modal.Volume` if needed for sharing trained models or datasets between runs/functions.

## 6. Data Handling
-   Input images/masks read from local paths provided via CLI.
-   Output images/models saved to local paths provided via CLI.
-   Intermediate data (like processed datasets or model checkpoints during training) might be stored temporarily in Modal or on a `modal.Volume`.
-   Path handling should be mindful of WSL/Linux differences (use `pathlib`).

## 7. Testing (`tests/`)
-   **Unit Tests**: Test individual functions (image utils, config validation).
-   **Integration Tests**: Test CLI command parsing, configuration loading. Mock Modal calls.
-   **Modal Tests**: Modal provides ways to test functions locally (`modal run --local ...`) which can be integrated into tests, though full end-to-end tests involving actual Modal runs might be manual or require specific setup.

## 8. Dependencies
-   Core: `python>=3.10`, `modal`, `uv`
-   ML/Torch: `torch`, `torchvision`, `torchaudio` (ensure CUDA compatibility if running locally)
-   Hugging Face: `diffusers`, `transformers`, `accelerate`, `peft`, `huggingface-hub`
-   CLI: `typer[all]` (includes `rich`)
-   Config: `pydantic`
-   Image: `Pillow` (or `opencv-python-headless`)
-   (Development): `pytest`, `ruff` 