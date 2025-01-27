from inference import VLMPipeline

pipeline = VLMPipeline(ModelConfig())
result = pipeline.generate(image, "What is in the image?")