from config.settings import ModelConfig
from inference.inference import VLMPipeline
from PIL import Image

if __name__ == "__main__":
    model_config = ModelConfig()
    model_path = "./results/pytorch_model.bin"  # Путь к сохраненным весам
    pipeline = VLMPipeline(model_config, model_path=model_path)

    image = Image.open("path_to_image.jpg")
    question = "What is in the image?"
    result = pipeline.generate(image, question)
    print(result)
