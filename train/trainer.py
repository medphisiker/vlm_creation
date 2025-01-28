from transformers import TrainingArguments, Trainer
from config.settings import TrainingConfig
import torch


def setup_training(config: TrainingConfig):
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=50,
        report_to=["tensorboard"],
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
    )


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = self._calculate_loss(outputs.logits, inputs["input_ids"])
        return (loss, outputs) if return_outputs else loss

    def _calculate_loss(self, logits, labels):
        # Учитываем смещение на 1 токен из-за конкатенации визуальных эмбеддингов
        shift_logits = logits[:, 1:-1, :]  # [batch, seq_len-2, vocab_size]
        shift_labels = labels[:, 1:]  # [batch, seq_len-1]

        # Изменяем размерности для вычисления потерь
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=self.model.llm.config.pad_token_id,
        )
        return loss
