# Image Finetuning and Inpainting CLI Tool

## Project Description
A command-line interface (CLI) tool designed for finetuning image generation models using Low-Rank Adaptation (LoRA) and subsequently using these (or other custom) models for advanced image inpainting tasks. The project aims to provide a flexible workflow for training models on custom image datasets and applying them creatively through inpainting with preprocessing options.

## Target Audience
- Machine Learning practitioners
- Digital artists
- Researchers working with generative models

## Desired Features
### Core Functionality
- [ ] LoRA Finetuning Module
  - [ ] Support for finetuning diffusion models (e.g., Flux, SDXL).
  - [ ] Handle large input images (~25MPix) for training via:
    - [ ] Resizing options.
    - [ ] Tiling strategies.
  - [ ] Configuration via CLI arguments or config files.
  - [ ] Integration with Modal for remote GPU execution.
- [ ] Image Inpainting Module
  - [ ] Utilize finetuned LoRA models or other custom diffusion models.
  - [ ] Input: Base image, mask image/definition.
  - [ ] Preprocessing options for the masked area before inpainting (e.g., convert to grayscale).
  - [ ] Postprocessing options for the masked area (e.g., revert preprocessing steps like grayscale).
  - [ ] Configuration via CLI arguments or config files.
  - [ ] Integration with Modal for remote GPU execution.
- [ ] CLI Interface
  - [ ] Commands for `finetune` and `inpaint` workflows.
  - [ ] Clear argument parsing and help messages.

## Technical Specifications
- [ ] Language: Python
- [ ] Project Management: `uv`
- [ ] Remote Execution/GPU: `Modal`
- [ ] Core Libraries: Hugging Face `diffusers`, `transformers`, `torch`
- [ ] Image Processing: `Pillow` or `opencv-python`

## Design Requests (CLI)
- [ ] Intuitive command structure (e.g., `python main.py finetune --config <path>`, `python main.py inpaint --model <path> --image <path> --mask <path> --preprocess grayscale`)
- [ ] Progress indicators for long-running tasks (training, inference).
- [ ] Clear logging and error reporting.

## Potential Future Enhancements
- [ ] GUI interface (e.g., Gradio, Streamlit).
- [ ] Support for more finetuning techniques beyond LoRA.
- [ ] More sophisticated preprocessing/postprocessing options for inpainting.
- [ ] Batch processing capabilities for finetuning and inpainting.
- [ ] Model management/versioning system.
- [ ] Integration with experiment tracking tools (e.g., Weights & Biases, MLflow). 