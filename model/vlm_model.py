import torch
from torch import nn
from transformers import AutoModelForCausalLM, ViTModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config.settings import ModelConfig


class VisionLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(
            config.vision_model, cache_dir=config.cache_dir, add_pooling_layer=False
        )
        self._freeze_vision_encoder()

        # Добавляем настройку pad_token_id для LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=config.cache_dir,
            pad_token_id=self._get_pad_token_id(config),
        )

        self.projection = self._build_projection(config)
        self._setup_lora(config)
        
    def _get_pad_token_id(self, config: ModelConfig):
        # Проверяем наличие pad_token_id у токенизатора
        tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model, 
            cache_dir=config.cache_dir
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer.pad_token_id

    def _freeze_vision_encoder(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def _build_projection(self, config: ModelConfig):
        # Для стандартной ViT берем hidden_size из конфига
        vision_dim = self.vision_encoder.config.hidden_size
        return nn.Sequential(
            nn.Linear(vision_dim, config.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(config.projection_hidden_dim, self.llm.config.hidden_size),
            nn.Dropout(0.1),
        )

    def _setup_lora(self, config: ModelConfig):
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            # Для ViT: output shape [batch_size, sequence_length, hidden_size]
            vision_output = self.vision_encoder(pixel_values)

            # Берем [CLS]-токен (первый токен в последовательности)
            visual_features = vision_output.last_hidden_state[:, 0, :]

        visual_embeds = self.projection(visual_features)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Конкатенация визуальных и текстовых эмбеддингов
        combined_embeds = torch.cat(
            [visual_embeds.unsqueeze(1), inputs_embeds],  # [batch_size, 1, hidden_size]
            dim=1,
        )

        # Расширяем маску внимания
        extended_attention_mask = torch.cat(
            [
                torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )

        return self.llm(
            inputs_embeds=combined_embeds, attention_mask=extended_attention_mask
        )
