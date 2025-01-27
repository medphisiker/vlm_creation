import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(vision_dim, llm_dim)

    def forward(self, visual_features):
        return self.linear(visual_features)

  
class MLPProjection(nn.Module):
    def __init__(self, vision_dim, llm_dim, hidden_dim=512):
        super(MLPProjection, self).__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.activation = nn.GELU()

    def forward(self, visual_features):
        x = self.activation(self.fc1(visual_features))
        return self.fc2(x)
    
from transformers import TransformerEncoder, TransformerEncoderLayer

class TransformerProjection(nn.Module):
    def __init__(self, vision_dim, llm_dim, num_layers=2, num_heads=8):
        super(TransformerProjection, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=vision_dim, nhead=num_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(vision_dim, llm_dim)

    def forward(self, visual_features):
        # Добавляем размер последовательности (batch_size, seq_len=1, vision_dim)
        visual_features = visual_features.unsqueeze(1)
        transformed = self.transformer(visual_features)
        return self.projection(transformed.squeeze(1))

class CrossAttentionProjection(nn.Module):
    def __init__(self, vision_dim, llm_dim, num_heads=8):
        super(CrossAttentionProjection, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads)
        self.projection = nn.Linear(vision_dim, llm_dim)

    def forward(self, visual_features, text_features):
        # Проекция визуальных признаков
        visual_projected = self.projection(visual_features)
        # Кросс-аттенция между текстовыми и визуальными признаками
        attn_output, _ = self.cross_attention(text_features, visual_projected, visual_projected)
        return attn_output