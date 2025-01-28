from transformers import TrainingArguments, Trainer
from config.settings import TrainingConfig


def setup_training(config: TrainingConfig):
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        cache_dir=config.cache_dir,
    )


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = self._calculate_loss(outputs.logits, inputs["input_ids"])
        return (loss, outputs) if return_outputs else loss

    def _calculate_loss(self, logits, labels):
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        return torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=self.model.llm.config.pad_token_id,
        )
