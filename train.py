import os
import aiohttp
from datasets import load_dataset
from config.settings import ModelConfig, TrainingConfig
from model.vlm_model import VisionLanguageModel
from data.processor import DataProcessor
from train.trainer import setup_training, CustomTrainer


def main():
    # Инициализация конфигов
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Создание папки кэша, если она не существует
    os.makedirs(ModelConfig.cache_dir, exist_ok=True)

    # Загрузка модели
    model = VisionLanguageModel(model_config)

    # Подготовка данных
    processor = DataProcessor(model_config)
    dataset = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="train[:1%]",
        cache_dir=ModelConfig.cache_dir,
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600*10)}
        },
    )
    # processed_dataset = dataset.map(
    #     lambda examples: {
    #         "pixel_values": processor.process_images(examples),
    #         **processor.process_text(examples),
    #     },
    #     batched=True,
    #     batch_size=32,
    # )

    # # Настройка обучения
    # training_args = setup_training(training_config)
    # trainer = CustomTrainer(
    #     model=model, args=training_args, train_dataset=processed_dataset
    # )

    # # Запуск обучения
    # trainer.train()


if __name__ == "__main__":
    main()
