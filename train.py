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

    # Создание папки кэша
    os.makedirs(model_config.cache_dir, exist_ok=True)

    # Загрузка модели
    model = VisionLanguageModel(model_config)

    # Подготовка данных
    processor = DataProcessor(model_config)
    
    # Загрузка тренировочных и валлидационных данных
    train_data = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="train[:1%]",
        cache_dir=model_config.cache_dir,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600*10)}}
    )
    
    eval_data = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="validation[:1%]",
        cache_dir=model_config.cache_dir,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600*10)}}
    )

    # Обработка данных
    def process_batch(examples):
        # Обработка изображений
        pixel_values = processor.process_images(examples)
        
        # Обработка текста
        text_data = processor.process_text(examples)
        
        # Возвращаем словарь с тензорами
        return {
            "pixel_values": pixel_values,
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
        }

    processed_train = train_data.map(
        process_batch,
        batched=True,
        batch_size=32,
        remove_columns=train_data.column_names,
    )
    
    processed_eval = eval_data.map(
        process_batch,
        batched=True,
        batch_size=32,
        remove_columns=eval_data.column_names,
    )

    # Обучение
    training_args = setup_training(training_config)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval
    )

    # Запуск обучения
    trainer.train()

if __name__ == "__main__":
    main()
