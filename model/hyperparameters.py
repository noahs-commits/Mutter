
from dataclasses import dataclass


@dataclass
class hyperparameters():
    transformer_dim: int
    layer_count: int
    head_count: int
    dim_feedforward: int
    activation: str
    dropout: float
    max_audio_len: int
    max_audio_lookback: int
    max_token: int
    n_tokens: int