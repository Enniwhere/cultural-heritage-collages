from dataclasses import dataclass

@dataclass
class BaseConfig:
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-dev"
    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 8
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117

@dataclass
class PostcardConfig(BaseConfig):
    # The instance name is the "proper noun" we're teaching the model
    instance_prompt: str = "A postcard of Aarhus the Danish city"

@dataclass
class AerialConfig(BaseConfig):
    # The instance name is the "proper noun" we're teaching the model
    instance_prompt: str = "An aerial view of Aarhus the Danish city"

