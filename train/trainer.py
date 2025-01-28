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
        evaluation_strategy="steps",
        eval_steps=config.eval_bleu_steps,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        predict_with_generate=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bleu = BLEU()
        self.best_bleu = 0.0
        
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
    
    def evaluation_step(self, model, inputs):
        # Генерация текста для вычисления BLEU
        generated_ids = model.generate(
            inputs["pixel_values"].to(model.device),
            input_ids=inputs["input_ids"][:, :1].to(model.device),  # Берем только BOS токен
            max_length=self.args.max_gen_length,
            num_beams=self.args.num_beams,
            early_stopping=True
        )
        return generated_ids, inputs["labels"]
    
    def compute_metrics(self, eval_preds):
        pred_ids, label_ids = eval_preds
        decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        # Декодируем метки, игнорируя pad_token
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Вычисляем BLEU score
        bleu_score = self.bleu.corpus_score(
            decoded_preds,
            [decoded_labels]
        ).score
        
        return {"bleu": bleu_score}

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
    
    def evaluate(self, **kwargs):
        # Сохраняем оригинальный токенизатор
        original_tokenizer = self.tokenizer
        self.tokenizer = self.data_collator.tokenizer
        
        eval_results = super().evaluate(**kwargs)
        
        # Восстанавливаем токенизатор
        self.tokenizer = original_tokenizer
        
        # Сохраняем лучшую модель
        if eval_results["bleu"] > self.best_bleu:
            self.best_bleu = eval_results["bleu"]
            self.save_model(f"{self.args.output_dir}/best_model")
            
        return eval_results
