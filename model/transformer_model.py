import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.gamma1 = nn.Parameter(torch.ones(hidden_dim))
        self.gamma2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        x_p = x.permute(1, 0, 2)
        attn_output, _ = self.self_attn(x_p, x_p, x_p)
        attn_output = attn_output.permute(1, 0, 2)
        x = x + self.dropout(self.gamma1 * self.norm1(attn_output))

        ffn_output = self.ffn(x)
        x = x + self.dropout(self.gamma2 * self.norm2(ffn_output))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.gamma1 = nn.Parameter(torch.ones(hidden_dim))
        self.gamma2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, query, key):
        assert query.dim() == 3 and key.dim(
        ) == 3, "Inputs must have 3 dimensions [batch_size, seq_len, hidden_dim]"

        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)

        attn_output, _ = self.cross_attn(query, query, key)
        attn_output = attn_output.permute(1, 0, 2)

        x = query.permute(1, 0, 2) + \
            self.dropout(self.gamma1 * self.norm1(attn_output))

        ffn_out = self.ffn(x)
        x = x + self.dropout(self.gamma2 * self.norm2(ffn_out))
        return x


class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, speech, text):
        combined = torch.cat([speech, text], dim=-1)
        fused = self.fusion_layer(combined)
        return self.norm(fused)


class FeatureProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.projection(x)


class SpeechTextTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes, num_layers=3):
        super().__init__()
        self.dropout_rate = 0.5

        self.speech_projection = FeatureProjection(
            input_dim, hidden_dim, self.dropout_rate)
        self.text_projection = FeatureProjection(
            input_dim, hidden_dim, self.dropout_rate)

        self.speech_conv_1 = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        self.text_conv_1 = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        self.speech_conv_2 = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        self.text_conv_2 = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        self.speech_conv_3 = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        self.text_conv_3 = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )
        self.speech_transformers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout_rate=self.dropout_rate)
             for _ in range(num_layers)]
        )
        self.text_transformers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout_rate=self.dropout_rate)
             for _ in range(num_layers)]
        )

        self.cross_attention_speech_to_text = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout_rate=self.dropout_rate)
             for _ in range(num_layers)]
        )
        self.cross_attention_text_to_speech = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout_rate=self.dropout_rate)
             for _ in range(num_layers)]
        )
        self.fusion = CrossModalFusion(
            hidden_dim, dropout_rate=self.dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, speech_features, text_features):
        speech = self.speech_projection(speech_features)
        text = self.text_projection(text_features)

        speech = speech.unsqueeze(1)
        text = text.unsqueeze(1)

        speech = self.speech_conv_1(speech)
        text = self.text_conv_1(text)

        identity_speech = speech
        identity_text = text

        for transformer in self.speech_transformers:
            speech = transformer(speech) + speech
        for transformer in self.text_transformers:
            text = transformer(text) + text

        speech = self.speech_conv_2(speech)
        text = self.text_conv_2(text)

        speech = speech + identity_speech
        text = text + identity_text

        speech_attended = 0
        text_attended = 0
        for crossattention in self.cross_attention_speech_to_text:
            speech_attended = crossattention(speech, text) + speech_attended

        for crossattention in self.cross_attention_text_to_speech:
            text_attended = crossattention(text, speech) + text_attended

        speech = speech + speech_attended
        text = text + text_attended

        speech = self.speech_conv_3(speech)
        text = self.text_conv_3(text)

        fused = self.fusion(speech.squeeze(1), text.squeeze(1))

        output = self.classifier(fused)
        return output
