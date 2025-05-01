import modal
from src.core.config import app, image, MODEL_DIR, volume
from dataclasses import dataclass

@dataclass
class InferenceConfig():
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 6

@app.cls(image=image, gpu="A100", volumes={MODEL_DIR: volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        # Reload the modal.Volume to ensure the latest state is accessible.
        volume.reload()

        # set up a hugging face inference pipeline using our model
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.load_lora_weights(MODEL_DIR)
        self.pipe = pipe

    @modal.method()
    def inference(self, text):
        config = InferenceConfig()
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


@app.local_entrypoint()
def main():
    model = Model()
    image = model.inference.remote("An aerial view of the Aarhus harbor")
    image.save("output.png")


