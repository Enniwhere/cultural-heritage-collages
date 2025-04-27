"""Pydantic models for configuration validation."""

from pydantic import BaseModel, Field, FilePath, DirectoryPath
from typing import List, Optional, Literal


class FinetuneConfig(BaseModel):
    base_model_id: str = Field(..., description="Hugging Face ID of the base model")
    dataset_dir: DirectoryPath = Field(..., description="Path to the input image directory")
    output_dir: DirectoryPath = Field(..., description="Path to save the trained LoRA weights")
    image_handling: Literal['resize', 'tile'] = Field('resize', description="How to handle large images")
    resolution: Optional[int] = Field(1024, description="Target resolution if resizing")
    tile_size: Optional[int] = Field(1024, description="Size of tiles if tiling")
    instance_prompt: str = Field(..., description="Prompt describing the subject (e.g., 'a photo of sks dog')")
    class_prompt: Optional[str] = Field(None, description="Prompt describing the class (e.g., 'a photo of a dog') - for prior preservation")
    max_train_steps: int = Field(500, description="Number of training steps")
    learning_rate: float = Field(1e-4, description="Training learning rate")
    lora_rank: int = Field(4, description="Rank for the LoRA matrices")
    batch_size: int = Field(1, alias="train_batch_size", description="Training batch size")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    # Add other diffusers/accelerate args as needed


class InpaintConfig(BaseModel):
    base_model_id: str = Field(..., description="Hugging Face ID of the base model")
    lora_weights_path: Optional[FilePath] = Field(None, description="Path to the trained LoRA weights")
    base_image_path: FilePath = Field(..., description="Path to the input image")
    mask_image_path: FilePath = Field(..., description="Path to the mask image (white areas indicate inpainting region)")
    output_path: FilePath = Field(..., description="Path to save the inpainted image")
    prompt: str = Field(..., description="Text prompt describing the desired inpainted content")
    negative_prompt: Optional[str] = Field(None, description="Text prompt for things to avoid")
    preprocess: List[Literal['grayscale']] = Field([], description="List of preprocessing steps")
    postprocess: List[Literal['revert_grayscale']] = Field([], description="List of postprocessing steps")
    num_inference_steps: int = Field(50, description="Number of diffusion steps")
    guidance_scale: float = Field(7.5, description="Guidance scale for inference")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    # TODO: Add validation for output_path directory existence?
    # TODO: Add validation for preprocess/postprocess combinations?

# Potential function to load config from file (YAML/JSON)
# def load_config(config_path: str) -> Union[FinetuneConfig, InpaintConfig]:
#     pass 