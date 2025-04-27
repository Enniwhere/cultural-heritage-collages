# Project Task List

## Phase 1: Project Setup & Core Structure
- [ ] Initialize project with `uv init`.
- [ ] Create basic directory structure (`src`, `data`, `tests`, `.notes`).
- [ ] Set up `pyproject.toml` with basic metadata and Python version (`>=3.10`).
- [ ] Add core dependencies (`modal`, `typer[all]`, `pydantic`, `pillow`) using `uv add`.
- [ ] Add Hugging Face dependencies (`diffusers`, `transformers`, `accelerate`, `peft`, `huggingface-hub`, `torch`, `torchvision`) using `uv add`.
- [ ] Add development dependencies (`pytest`, `ruff`) using `uv add -G dev`.
- [ ] Configure `ruff` via `pyproject.toml` or `ruff.toml`.
- [ ] Create initial `README.md`.
- [ ] Set up basic `.gitignore`.
- [ ] Perform initial `uv lock` and `uv sync`.
- [ ] Commit initial project structure.

## Phase 2: CLI Foundation
- [ ] Implement basic CLI entry point in `src/cli/main.py` using Typer.
- [ ] Define `finetune` command structure in `src/cli/commands/finetune.py`.
- [ ] Define `inpaint` command structure in `src/cli/commands/inpaint.py`.
- [ ] Set up Pydantic models for `FinetuneConfig` and `InpaintConfig` in `src/config/models.py`.
- [ ] Integrate basic config loading/validation into CLI commands (reading args, potential config file support).
- [ ] Add basic logging setup.

## Phase 3: Modal Setup
- [ ] Define `modal.App` in `src/core/modal_setup.py`.
- [ ] Create base `modal.Image` with Python & system dependencies.
- [ ] Define `modal.Secret` for Hugging Face token.
- [ ] Define `modal.Volume` for potential model/data sharing (e.g., `model-cache-volume`).
- [ ] Test basic Modal connectivity (`modal run src/core/modal_setup.py` with a dummy function).

## Phase 4: Finetuning Implementation
- [ ] Implement image handling utilities (`resize`, `tile`) in `src/core/finetuning/data_utils.py`.
- [ ] Create the Modal finetuning function/class structure in `src/core/finetuning/training.py`.
- [ ] Adapt or integrate `diffusers` LoRA training script logic within the Modal function.
- [ ] Implement data loading within the Modal function (handle local paths, potentially mount `data/input_images`).
- [ ] Implement logic to run `accelerate launch` within Modal.
- [ ] Handle saving LoRA weights to the specified `output_dir` (potentially via Volume or downloading from Modal).
- [ ] Connect `finetune` CLI command to trigger the Modal function, passing the validated `FinetuneConfig`.
- [ ] Add progress reporting for the Modal job in the CLI.

## Phase 5: Inpainting Implementation
- [ ] Implement image preprocessing/postprocessing utilities (`grayscale`, `revert_grayscale`, mask handling) in `src/core/inpainting/image_utils.py`.
- [ ] Create the Modal inference class structure (`@modal.cls`) in `src/core/inpainting/inference.py`.
- [ ] Implement model loading (`DiffusionPipeline`, LoRA weights) in `@modal.enter`.
- [ ] Implement inference logic in `@modal.method`, including:
    - [ ] Loading base image and mask.
    - [ ] Applying preprocessing.
    - [ ] Running the inpainting pipeline.
    - [ ] Applying postprocessing.
    - [ ] Saving the output image.
- [ ] Connect `inpaint` CLI command to trigger the Modal method, passing the validated `InpaintConfig`.
- [ ] Add progress reporting for the Modal job in the CLI.

## Phase 6: Testing & Refinement
- [ ] Write unit tests for `data_utils.py` and `image_utils.py`.
- [ ] Write unit tests for config validation (`src/config/models.py`).
- [ ] Write integration tests for CLI argument parsing and command invocation (mocking Modal calls).
- [ ] Perform manual end-to-end tests for both `finetune` and `inpaint` workflows.
- [ ] Refine CLI output, logging, and error handling.
- [ ] Update `README.md` with detailed usage instructions.
- [ ] Run `ruff check . --fix` and `ruff format .`.

## Phase 7: Documentation & Cleanup
- [ ] Add docstrings to major functions and classes.
- [ ] Review and finalize `README.md`.
- [ ] Clean up any temporary files or unused code.
- [ ] Ensure `uv.lock` is up-to-date and committed. 