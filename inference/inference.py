import torch
from PIL import Image
from config.settings import ModelConfig
from model.vlm_model import VisionLanguageModel
from data.processor import DataProcessor

class VLMPipeline:
    def __init__(self, model_config: ModelConfig):
        self.model = VisionLanguageModel(model_config)
        self.processor = DataProcessor(model_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate(self, image: Image, question: str, max_length: int = 200):
        pixel_values = self.processor.process_images({"image": [image]})[0].unsqueeze(0).to(self.device)
        inputs = self.processor.process_text({
            "question": [question],
            "answers": [""]
        }).to(self.device)
        
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5
        )
        
        return self.processor.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        ).split("### Answer: ")[-1]