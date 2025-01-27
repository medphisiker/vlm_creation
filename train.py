from datasets import load_dataset
from config.settings import ModelConfig, TrainingConfig
from model.vlm_model import VisionLanguageModel
from data.processor import DataProcessor
from training.trainer import setup_training, CustomTrainer


def main():
    # Инициализация конфигов
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Загрузка модели
    model = VisionLanguageModel(model_config)

    # Подготовка данных
    processor = DataProcessor(model_config)
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train[:10%]")

    processed_dataset = dataset.map(
        lambda examples: {
            "pixel_values": processor.process_images(examples),
            **processor.process_text(examples),
        },
        batched=True,
        batch_size=32,
    )

    # Настройка обучения
    training_args = setup_training(training_config)
    trainer = CustomTrainer(
        model=model, args=training_args, train_dataset=processed_dataset
    )

    # Запуск обучения
    trainer.train()


if __name__ == "__main__":
    main()
