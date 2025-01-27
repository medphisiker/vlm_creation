import torch
from torch import nn
from transformers import AutoModelForCausalLM, MobileViTModel
from peft import LoraConfig, get_peft_model
from config.settings import ModelConfig

class VisionLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vision_encoder = MobileViTModel.from_pretrained(config.vision_model)
        self._freeze_vision_encoder()
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        self.projection = self._build_projection(config)
        self._setup_lora(config)

    def _freeze_vision_encoder(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def _build_projection(self, config: ModelConfig):
        vision_dim = self.vision_encoder.config.hidden_sizes[-1]
        return nn.Sequential(
            nn.Linear(vision_dim, config.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(config.projection_hidden_dim, self.llm.config.hidden_size),
            nn.Dropout(0.1)
        )

    def _setup_lora(self, config: ModelConfig):
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            visual_features = self.vision_encoder(pixel_values).last_hidden_state.mean(1)
        
        visual_embeds = self.projection(visual_features)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        combined_embeds = torch.cat([
            visual_embeds.unsqueeze(1),
            inputs_embeds
        ], dim=1)
        
        extended_attention_mask = torch.cat([
            torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
            attention_mask
        ], dim=1)
        
        return self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=extended_attention_mask
        )