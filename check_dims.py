from transformers import AutoModelForCausalLM, MobileViTModel

if __name__ == "__main__":
    
    # Указываем папку для кэширования
    cache_dir = "model_cache"

    # Модели будут загружены в указанную папку
    vision_encoder = MobileViTModel.from_pretrained("apple/mobilevit-small", cache_dir=cache_dir)
    llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir=cache_dir)

    print("Vision encoder output dim:", vision_encoder.config.hidden_sizes[-1])
    print("LLM input dim:", llm.config.hidden_size)
