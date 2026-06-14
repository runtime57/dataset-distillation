import torch
from torch import nn

from src.distillation.parameterizations.base import BaseSyntheticTokenDataset


def _sample_unique_support(
    num_positions,
    vocab_size,
    k,
    device,
    forced_ids=None,
    chunk_size=4096,
):
    support = torch.empty(num_positions, k, dtype=torch.long, device=device)
    for start in range(0, num_positions, chunk_size):
        end = min(start + chunk_size, num_positions)
        scores = torch.rand(end - start, vocab_size, device=device)
        if forced_ids is not None:
            scores.scatter_(1, forced_ids[start:end].unsqueeze(1), 2.0)
        support[start:end] = scores.topk(k, dim=-1).indices
    return support


class FullSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Trainable full soft-token representation for synthetic text.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, vocab_size) * init_std
        )

    def token_probs(self, indices=None, embedding_weight=None):
        logits = self.logits if indices is None else self.logits[indices]
        return torch.softmax(logits / self.temperature, dim=-1)

    def initialize_from_token_ids(
        self,
        input_ids,
        confidence=5.0,
        embedding_weight=None,
    ):
        if input_ids.shape != (self.num_sequences, self.sequence_length):
            raise ValueError(
                "Expected input_ids shape "
                f"{(self.num_sequences, self.sequence_length)}, got "
                f"{tuple(input_ids.shape)}."
            )
        with torch.no_grad():
            self.logits.zero_()
            self.logits.scatter_(-1, input_ids.unsqueeze(-1), confidence)
            self.logits.add_(torch.randn_like(self.logits) * 0.01)

    def initialize_from_token_probs(
        self,
        token_probs,
        confidence=5.0,
        embedding_weight=None,
    ):
        if token_probs.shape != (self.num_sequences, self.sequence_length, self.vocab_size):
            raise ValueError(
                "Expected token_probs shape "
                f"{(self.num_sequences, self.sequence_length, self.vocab_size)}, got "
                f"{tuple(token_probs.shape)}."
            )
        with torch.no_grad():
            probs = token_probs / token_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            self.logits.copy_(torch.log(probs.clamp_min(1e-12)) * self.temperature)


class TopKSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Sparse top-k vocabulary distribution per synthetic token position.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        k,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if k < 1:
            raise ValueError(f"top-k support size must be positive, got {k}.")
        if k > vocab_size:
            raise ValueError(
                f"top-k support size {k} exceeds vocab size {vocab_size}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, k) * init_std
        )
        initial_support = _sample_unique_support(
            num_positions=num_sequences * sequence_length,
            vocab_size=vocab_size,
            k=k,
            device=torch.device("cpu"),
        ).view(num_sequences, sequence_length, k)
        self.register_buffer("support_ids", initial_support)

    def token_probs(self, indices=None, embedding_weight=None):
        logits = self.logits if indices is None else self.logits[indices]
        support_ids = self.support_ids if indices is None else self.support_ids[indices]
        weights = torch.softmax(logits / self.temperature, dim=-1)
        probs = torch.zeros(
            *weights.shape[:-1],
            self.vocab_size,
            dtype=weights.dtype,
            device=weights.device,
        )
        return probs.scatter_add(-1, support_ids, weights)

    def initialize_from_token_ids(
        self,
        input_ids,
        confidence=5.0,
        embedding_weight=None,
    ):
        if input_ids.shape != (self.num_sequences, self.sequence_length):
            raise ValueError(
                "Expected input_ids shape "
                f"{(self.num_sequences, self.sequence_length)}, got "
                f"{tuple(input_ids.shape)}."
            )
        with torch.no_grad():
            support_ids = _sample_unique_support(
                num_positions=self.num_sequences * self.sequence_length,
                vocab_size=self.vocab_size,
                k=self.k,
                device=input_ids.device,
                forced_ids=input_ids.reshape(-1),
            ).view(self.num_sequences, self.sequence_length, self.k)
            self.support_ids.copy_(support_ids)
            self.logits.zero_()
            self.logits[..., 0] = confidence
            self.logits.add_(torch.randn_like(self.logits) * 0.01)


class GumbelTopKSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Full logits with Gumbel-top-k sparse forward pass.

    Support is learned: it tracks the top-k tokens by logit value and shifts
    during training. Gumbel noise provides stochastic exploration of different
    support sets across steps. Backward uses a straight-through estimator:
    gradients flow through the full softmax, not through the discrete selection.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        k,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, vocab_size) * init_std
        )

    def token_probs(self, indices=None, embedding_weight=None):
        logits = self.logits if indices is None else self.logits[indices]
        full_probs = torch.softmax(logits / self.temperature, dim=-1)

        if self.training:
            uniform = torch.rand_like(logits).clamp(min=1e-20)
            gumbel = -torch.log(-torch.log(uniform))
            selection_scores = (logits + gumbel).detach()
        else:
            selection_scores = logits.detach()

        topk_indices = selection_scores.topk(self.k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)

        masked_logits = logits.masked_fill(~mask, float("-inf"))
        sparse_probs = torch.softmax(masked_logits / self.temperature, dim=-1)
        return sparse_probs.detach() + full_probs - full_probs.detach()

    def initialize_from_token_ids(
        self,
        input_ids,
        confidence=5.0,
        embedding_weight=None,
    ):
        if input_ids.shape != (self.num_sequences, self.sequence_length):
            raise ValueError(
                "Expected input_ids shape "
                f"{(self.num_sequences, self.sequence_length)}, got "
                f"{tuple(input_ids.shape)}."
            )
        with torch.no_grad():
            self.logits.zero_()
            self.logits.scatter_(-1, input_ids.unsqueeze(-1), confidence)
            self.logits.add_(torch.randn_like(self.logits) * 0.01)

    def initialize_from_token_probs(
        self,
        token_probs,
        confidence=5.0,
        embedding_weight=None,
    ):
        if token_probs.shape != (self.num_sequences, self.sequence_length, self.vocab_size):
            raise ValueError(
                "Expected token_probs shape "
                f"{(self.num_sequences, self.sequence_length, self.vocab_size)}, got "
                f"{tuple(token_probs.shape)}."
            )
        with torch.no_grad():
            probs = token_probs / token_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            self.logits.copy_(torch.log(probs.clamp_min(1e-12)) * self.temperature)
