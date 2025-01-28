from dataclasses import dataclass

@dataclass
class ModelConfig:
    vision_model: str = "google/vit-base-patch16-224-in21k"
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_rank: int = 8
    projection_hidden_dim: int = 1024
    cache_dir: str = "./model_cache"

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 512
    eval_bleu_steps: int = 500
    num_beams: int = 3
    max_gen_length: int = 50
    output_dir: str = "./results"