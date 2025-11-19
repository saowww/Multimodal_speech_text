import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    def selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=None, delta_softplus=False):
        return u


class MambaScan1D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, expand_factor=2, dropout_rate=0.5):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(
            self.d_model / 16) if dt_rank is None else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )

        self.batch_norm = nn.BatchNorm1d(self.d_inner)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        dt_min_log = math.log(dt_min)
        dt_max_log = math.log(dt_max)

        if dt_init == "random":
            init_values = torch.exp(torch.rand(
                self.d_inner) * (dt_max_log - dt_min_log) + dt_min_log)
            with torch.no_grad():
                self.dt_proj.weight.zero_()
                self.dt_proj.bias.copy_(init_values)
        else:
            init_value = torch.exp(torch.tensor(
                0.5 * (dt_max_log - dt_min_log) + dt_min_log))
            with torch.no_grad():
                self.dt_proj.weight.zero_()
                self.dt_proj.bias.fill_(init_value.item())

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        identity = x
        x = x.to(device)

        x_norm = self.norm(x)

        x_proj = self.in_proj(x_norm)
        z, x_proj = x_proj.chunk(2, dim=-1)
        z = F.silu(z)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = self.batch_norm(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        delta = self.dt_proj(x_conv)

        u = x_conv.transpose(1, 2)
        delta = delta.transpose(1, 2)

        A = -torch.exp(self.A_log)
        B = self.B
        C = self.C
        D = self.D

        try:
            y = selective_scan_fn(
                u=u,
                delta=delta,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z.transpose(1, 2),
                delta_bias=self.dt_bias,
                delta_softplus=True
            )
        except Exception:
            u = u.transpose(1, 2)
            delta = delta.transpose(1, 2)

            delta = delta.unsqueeze(-1)
            A_discrete = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta)

            B = B.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_len, -1, -1)
            C = C.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_len, -1, -1)

            B = B * delta

            u = u.unsqueeze(-1)

            h = torch.zeros(batch_size, self.d_inner,
                            self.d_state, device=x.device)

            ys = []

            for t in range(seq_len):
                h = A_discrete[:, t] * h + u[:, t] * B[:, t]

                y_t = torch.sum(h * C[:, t], dim=-1) + self.D * u[:, t, :, 0]
                ys.append(y_t)

            y = torch.stack(ys, dim=1)

            y = y * z

            y = y.transpose(1, 2)

        if y.shape[1] != seq_len:
            y = y.transpose(1, 2)

        y = self.out_proj(y)

        return y + identity


class FeatureProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        output = self.projection(x)
        return output


class MambaCrossModalBlock(nn.Module):
    def __init__(self, hidden_dim, d_state=16, expand_factor=2, dropout_rate=0.6):
        super().__init__()

        self.audio_mamba = MambaScan1D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand_factor,
            dropout_rate=dropout_rate
        )

        self.text_mamba = MambaScan1D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand_factor,
            dropout_rate=dropout_rate
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.joint_mamba = MambaScan1D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand_factor,
            dropout_rate=dropout_rate
        )

        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, audio_features, text_features):
        audio_seq = audio_features.unsqueeze(1)
        text_seq = text_features.unsqueeze(1)

        audio_processed = self.audio_mamba(audio_seq).squeeze(1)
        text_processed = self.text_mamba(text_seq).squeeze(1)

        audio_importance = self.audio_gate(audio_processed)
        text_importance = self.text_gate(text_processed)

        audio_weighted = audio_processed * audio_importance
        text_weighted = text_processed * text_importance

        combined_features = torch.cat([audio_weighted, text_weighted], dim=-1)

        fused_features = self.fusion_layer(combined_features)

        fused_seq = fused_features.unsqueeze(1)

        processed_features = self.joint_mamba(fused_seq)

        processed_features = processed_features.squeeze(1)

        processed_features = self.norm(processed_features)
        processed_features = self.dropout(processed_features)

        return processed_features


class MambaEmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, d_state=16, num_classes=5, dropout_rate=0.6):
        super().__init__()

        self.audio_projection = FeatureProjection(
            input_dim, hidden_dim, dropout_rate)
        self.text_projection = FeatureProjection(
            input_dim, hidden_dim, dropout_rate)

        self.cross_modal_block = MambaCrossModalBlock(
            hidden_dim=hidden_dim,
            d_state=d_state,
            expand_factor=2,
            dropout_rate=dropout_rate
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.2),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.3),
            nn.Linear(hidden_dim // 8, num_classes)
        )

    def forward(self, audio_features, text_features, temperature=1.0):
        audio_projected = self.audio_projection(audio_features)
        text_projected = self.text_projection(text_features)

        combined_features = self.cross_modal_block(
            audio_projected, text_projected)

        logits = self.classifier(combined_features) / temperature

        return logits
