import math

import torch
from torch import nn

from src.distillation.parameterizations.base import (
    BaseSyntheticTokenDataset,
    _kmeans,
)


class AnchorSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Shared soft-token anchors mixed at each synthetic token position.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        num_anchors,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if num_anchors < 1:
            raise ValueError(
                f"Number of anchors must be positive, got {num_anchors}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_anchors = num_anchors
        self.temperature = temperature
        self.anchor_logits = nn.Parameter(
            torch.randn(num_anchors, vocab_size) * init_std
        )
        self.mixture_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, num_anchors) * init_std
        )

    def token_probs(self, indices=None, embedding_weight=None):
        mixture_logits = (
            self.mixture_logits if indices is None else self.mixture_logits[indices]
        )
        anchors = torch.softmax(self.anchor_logits / self.temperature, dim=-1)
        mixtures = torch.softmax(mixture_logits / self.temperature, dim=-1)
        return torch.einsum("...k,kv->...v", mixtures, anchors)

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
        flat_tokens = input_ids.reshape(-1)
        flat_anchor_ids = torch.arange(flat_tokens.numel(), device=input_ids.device)
        flat_anchor_ids = flat_anchor_ids.remainder(self.num_anchors)
        with torch.no_grad():
            self.anchor_logits.zero_()
            anchor_tokens = flat_tokens[: self.num_anchors]
            if anchor_tokens.numel() < self.num_anchors:
                repeat_count = math.ceil(self.num_anchors / flat_tokens.numel())
                anchor_tokens = flat_tokens.repeat(repeat_count)[: self.num_anchors]
            self.anchor_logits.scatter_(
                -1,
                anchor_tokens[: self.num_anchors].unsqueeze(-1),
                confidence,
            )
            self.anchor_logits.add_(torch.randn_like(self.anchor_logits) * 0.01)

            self.mixture_logits.zero_()
            self.mixture_logits.view(-1, self.num_anchors).scatter_(
                -1,
                flat_anchor_ids.unsqueeze(-1),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)

    def initialize_from_kmeans(self, embedding_weight, confidence=5.0, n_iter=100):
        """Initialize anchors via K-means for diverse vocabulary coverage."""
        with torch.no_grad():
            centers = _kmeans(
                embedding_weight.detach(),
                k=self.num_anchors,
                n_iter=n_iter,
            )
            dists = torch.cdist(centers, embedding_weight.detach().float())
            nearest_tokens = dists.argmin(dim=-1)
            self.anchor_logits.zero_()
            self.anchor_logits.scatter_(-1, nearest_tokens.unsqueeze(-1), confidence)
            self.anchor_logits.add_(torch.randn_like(self.anchor_logits) * 0.01)

            total_positions = self.num_sequences * self.sequence_length
            flat_anchor_ids = torch.arange(
                total_positions,
                device=embedding_weight.device,
            )
            flat_anchor_ids = flat_anchor_ids.remainder(self.num_anchors)
            self.mixture_logits.zero_()
            self.mixture_logits.view(-1, self.num_anchors).scatter_(
                -1,
                flat_anchor_ids.unsqueeze(-1),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)


class SequenceAnchorSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Per-sequence soft-token anchors mixed at each token position.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        num_anchors,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if num_anchors < 1:
            raise ValueError(
                f"Number of anchors must be positive, got {num_anchors}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_anchors = num_anchors
        self.temperature = temperature
        self.anchor_logits = nn.Parameter(
            torch.randn(num_sequences, num_anchors, vocab_size) * init_std
        )
        self.mixture_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, num_anchors) * init_std
        )

    def token_probs(self, indices=None, embedding_weight=None):
        mixture_logits = (
            self.mixture_logits if indices is None else self.mixture_logits[indices]
        )
        anchor_logits = (
            self.anchor_logits if indices is None else self.anchor_logits[indices]
        )
        anchors = torch.softmax(anchor_logits / self.temperature, dim=-1)
        mixtures = torch.softmax(mixture_logits / self.temperature, dim=-1)
        return torch.einsum("...tk,...kv->...tv", mixtures, anchors)

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
        device = input_ids.device
        source_positions = torch.div(
            torch.arange(self.num_anchors, device=device) * self.sequence_length,
            self.num_anchors,
            rounding_mode="floor",
        ).clamp(max=self.sequence_length - 1)
        anchor_tokens = input_ids[:, source_positions]
        position_anchor_ids = torch.div(
            torch.arange(self.sequence_length, device=device) * self.num_anchors,
            self.sequence_length,
            rounding_mode="floor",
        ).clamp(max=self.num_anchors - 1)
        with torch.no_grad():
            self.anchor_logits.zero_()
            self.anchor_logits.scatter_(-1, anchor_tokens.unsqueeze(-1), confidence)
            self.anchor_logits.add_(torch.randn_like(self.anchor_logits) * 0.01)

            self.mixture_logits.zero_()
            self.mixture_logits.scatter_(
                -1,
                position_anchor_ids.view(1, self.sequence_length, 1).expand(
                    self.num_sequences, -1, -1
                ),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)


class SparseSequenceAnchorSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Per-sequence anchors with sparse vocabulary support.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        num_anchors,
        anchor_topk,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if num_anchors < 1:
            raise ValueError(
                f"Number of anchors must be positive, got {num_anchors}."
            )
        if anchor_topk < 1:
            raise ValueError(
                f"Anchor top-k support size must be positive, got {anchor_topk}."
            )
        if anchor_topk > vocab_size:
            raise ValueError(
                f"Anchor top-k support size {anchor_topk} exceeds vocab size "
                f"{vocab_size}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_anchors = num_anchors
        self.anchor_topk = anchor_topk
        self.temperature = temperature
        self.anchor_logits = nn.Parameter(
            torch.randn(num_sequences, num_anchors, anchor_topk) * init_std
        )
        self.mixture_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, num_anchors) * init_std
        )
        self.register_buffer(
            "support_ids",
            torch.randint(
                vocab_size,
                size=(num_sequences, num_anchors, anchor_topk),
            ),
        )

    def token_probs(self, indices=None, embedding_weight=None):
        mixture_logits = (
            self.mixture_logits if indices is None else self.mixture_logits[indices]
        )
        anchor_logits = (
            self.anchor_logits if indices is None else self.anchor_logits[indices]
        )
        support_ids = self.support_ids if indices is None else self.support_ids[indices]
        anchors = torch.softmax(anchor_logits / self.temperature, dim=-1)
        mixtures = torch.softmax(mixture_logits / self.temperature, dim=-1)
        weights = mixtures.unsqueeze(-1) * anchors.unsqueeze(-3)
        weights = weights.reshape(
            *mixtures.shape[:-1],
            self.num_anchors * self.anchor_topk,
        )
        ids = support_ids.unsqueeze(-3).expand(
            *mixtures.shape[:-1],
            self.num_anchors,
            self.anchor_topk,
        )
        ids = ids.reshape(*mixtures.shape[:-1], self.num_anchors * self.anchor_topk)
        probs = torch.zeros(
            *mixtures.shape[:-1],
            self.vocab_size,
            dtype=weights.dtype,
            device=weights.device,
        )
        return probs.scatter_add(-1, ids, weights)

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
        device = input_ids.device
        anchor_ids = torch.arange(self.num_anchors, device=device)
        support_offsets = torch.arange(self.anchor_topk, device=device)
        start_positions = torch.div(
            anchor_ids * self.sequence_length,
            self.num_anchors,
            rounding_mode="floor",
        )
        support_positions = (
            start_positions.unsqueeze(-1) + support_offsets.unsqueeze(0)
        ).remainder(self.sequence_length)
        position_anchor_ids = torch.div(
            torch.arange(self.sequence_length, device=device) * self.num_anchors,
            self.sequence_length,
            rounding_mode="floor",
        ).clamp(max=self.num_anchors - 1)
        covered_positions = math.ceil(self.sequence_length / self.num_anchors)
        covered_positions = min(covered_positions, self.anchor_topk)
        with torch.no_grad():
            self.support_ids.copy_(input_ids[:, support_positions])

            self.anchor_logits.zero_()
            self.anchor_logits[..., :covered_positions] = confidence
            self.anchor_logits.add_(torch.randn_like(self.anchor_logits) * 0.01)

            self.mixture_logits.zero_()
            self.mixture_logits.scatter_(
                -1,
                position_anchor_ids.view(1, self.sequence_length, 1).expand(
                    self.num_sequences, -1, -1
                ),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)


class GroupedAnchorSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Grouped soft-token anchors mixed at each synthetic token position.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        num_anchors,
        num_groups,
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if num_anchors < 1:
            raise ValueError(
                f"Number of anchors must be positive, got {num_anchors}."
            )
        if num_groups < 1:
            raise ValueError(f"Number of groups must be positive, got {num_groups}.")
        if num_groups > num_sequences:
            raise ValueError(
                f"Number of groups {num_groups} exceeds sequences {num_sequences}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_anchors = num_anchors
        self.num_groups = num_groups
        self.temperature = temperature
        self.anchor_logits = nn.Parameter(
            torch.randn(num_groups, num_anchors, vocab_size) * init_std
        )
        self.mixture_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, num_anchors) * init_std
        )
        group_ids = torch.arange(num_sequences).remainder(num_groups)
        self.register_buffer("group_ids", group_ids, persistent=False)

    def token_probs(self, indices=None, embedding_weight=None):
        mixture_logits = (
            self.mixture_logits if indices is None else self.mixture_logits[indices]
        )
        group_ids = self.group_ids if indices is None else self.group_ids[indices]
        anchor_logits = self.anchor_logits[group_ids]
        anchors = torch.softmax(anchor_logits / self.temperature, dim=-1)
        mixtures = torch.softmax(mixture_logits / self.temperature, dim=-1)
        return torch.einsum("...tk,...kv->...tv", mixtures, anchors)

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
        device = input_ids.device
        position_anchor_ids = torch.arange(self.sequence_length, device=device)
        position_anchor_ids = position_anchor_ids.remainder(self.num_anchors)
        flat_tokens = input_ids.reshape(-1)
        with torch.no_grad():
            self.anchor_logits.zero_()
            for group_id in range(self.num_groups):
                seq_indices = (self.group_ids == group_id).nonzero(as_tuple=True)[0]
                group_tokens = input_ids[seq_indices].reshape(-1)
                if group_tokens.numel() == 0:
                    group_tokens = flat_tokens
                if group_tokens.numel() < self.num_anchors:
                    repeat_count = math.ceil(self.num_anchors / group_tokens.numel())
                    group_tokens = group_tokens.repeat(repeat_count)
                anchor_tokens = group_tokens[: self.num_anchors]
                self.anchor_logits[group_id].scatter_(
                    -1,
                    anchor_tokens.unsqueeze(-1),
                    confidence,
                )
            self.anchor_logits.add_(torch.randn_like(self.anchor_logits) * 0.01)

            self.mixture_logits.zero_()
            self.mixture_logits.scatter_(
                -1,
                position_anchor_ids.view(1, self.sequence_length, 1).expand(
                    self.num_sequences, -1, -1
                ),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)
