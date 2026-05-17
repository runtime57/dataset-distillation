import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, attention_mask=None):
        batch_size, seq_len, d_model = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(
                key_padding_mask,
                torch.finfo(scores.dtype).min,
            )

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.output_projection(output)

    def _split_heads(self, tensor):
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return tensor.transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden, attention_mask=None):
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
        )
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden


class TinyTransformerLM(nn.Module):
    """
    Small causal Transformer language model for quick distillation experiments.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self._init_parameters()

    def forward(
        self,
        input_ids=None,
        input_embeds=None,
        target_probs=None,
        attention_mask=None,
        return_hidden_states=False,
        **batch,
    ):
        if input_embeds is None:
            if target_probs is not None:
                input_embeds = target_probs @ self.token_embedding.weight
            elif input_ids is not None:
                input_embeds = self.token_embedding(input_ids)
            else:
                raise ValueError(
                    "TinyTransformerLM expects input_ids, target_probs, or input_embeds."
                )

        batch_size, seq_len, _ = input_embeds.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        positions = torch.arange(seq_len, device=input_embeds.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        hidden = input_embeds * math.sqrt(input_embeds.shape[-1])
        hidden = hidden + self.position_embedding(positions)

        hidden_states = []
        for layer in self.transformer:
            hidden = layer(hidden, attention_mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(hidden)
        logits = self.output_projection(self.norm(hidden))
        result = {"logits": logits}
        if return_hidden_states:
            result["hidden_states"] = hidden_states
        return result

    def get_input_embeddings(self):
        return self.token_embedding

    def _init_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
