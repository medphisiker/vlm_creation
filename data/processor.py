from transformers import AutoTokenizer, MobileViTImageProcessor


class DataProcessor:
    def __init__(self, model_config):
        self.image_processor = MobileViTImageProcessor.from_pretrained(
            model_config.vision_model, cache_dir=model_config.cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.llm_model, cache_dir=model_config.cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_text(self, examples):
        return self.tokenizer(
            [
                f"### Question: {q}\n ### Answer: {a}"
                for q, a in zip(examples["question"], examples["answers"])
            ],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def process_images(self, examples):
        images = [img.convert("RGB") for img in examples["image"]]
        return self.image_processor(images, return_tensors="pt").pixel_values
