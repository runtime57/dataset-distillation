import math

import torch
from torch import nn

from src.distillation.parameterizations.base import (
    BaseSyntheticTokenDataset,
    _kmeans,
)


class ConceptSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Compact concept vectors mixed in embedding space and projected to tokens.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        num_concepts,
        d_model,
        logit_scale=32.0,
        input_mode="probs",
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if num_concepts < 1:
            raise ValueError(
                f"Number of concepts must be positive, got {num_concepts}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_concepts = num_concepts
        self.d_model = d_model
        self.logit_scale = float(logit_scale)
        if input_mode not in {"probs", "concepts"}:
            raise ValueError(
                "concept input_mode must be either 'probs' or 'concepts', "
                f"got {input_mode!r}."
            )
        self.input_mode = input_mode
        self.temperature = temperature
        self.concept_vectors = nn.Parameter(
            torch.randn(num_concepts, d_model) * init_std
        )
        self.mixture_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, num_concepts) * init_std
        )

    def concept_mixtures(self, indices=None):
        logits = self.mixture_logits if indices is None else self.mixture_logits[indices]
        return torch.softmax(logits / self.temperature, dim=-1)

    def concept_input_embeds(self, indices=None):
        return self.concept_mixtures(indices) @ self.concept_vectors

    def input_embeds(self, indices, embedding_weight):
        if self.input_mode == "concepts":
            return self.concept_input_embeds(indices)
        return self.token_probs(indices, embedding_weight) @ embedding_weight

    def token_probs(self, indices=None, embedding_weight=None):
        if embedding_weight is None:
            raise ValueError("ConceptSoftTokenDataset requires embedding_weight.")
        input_embeds = self.concept_input_embeds(indices)
        logits = input_embeds @ embedding_weight.transpose(0, 1)
        logits = logits * self.logit_scale
        return torch.softmax(logits / self.temperature, dim=-1)

    def initialize_from_token_ids(
        self,
        input_ids,
        confidence=5.0,
        embedding_weight=None,
    ):
        if embedding_weight is None:
            raise ValueError(
                "ConceptSoftTokenDataset initialization requires embedding_weight."
            )
        if input_ids.shape != (self.num_sequences, self.sequence_length):
            raise ValueError(
                "Expected input_ids shape "
                f"{(self.num_sequences, self.sequence_length)}, got "
                f"{tuple(input_ids.shape)}."
            )
        flat_tokens = input_ids.reshape(-1)
        flat_concept_ids = torch.arange(flat_tokens.numel(), device=input_ids.device)
        flat_concept_ids = flat_concept_ids.remainder(self.num_concepts)
        with torch.no_grad():
            concept_tokens = flat_tokens[: self.num_concepts]
            if concept_tokens.numel() < self.num_concepts:
                repeat_count = math.ceil(self.num_concepts / flat_tokens.numel())
                concept_tokens = flat_tokens.repeat(repeat_count)[: self.num_concepts]
            self.concept_vectors.copy_(
                embedding_weight[concept_tokens[: self.num_concepts]]
            )
            self.concept_vectors.add_(torch.randn_like(self.concept_vectors) * 0.01)

            self.mixture_logits.zero_()
            self.mixture_logits.view(-1, self.num_concepts).scatter_(
                -1,
                flat_concept_ids.unsqueeze(-1),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)

    def initialize_from_kmeans(self, embedding_weight, confidence=5.0, n_iter=100):
        """Initialize concept vectors via K-means on token embeddings."""
        with torch.no_grad():
            centers = _kmeans(
                embedding_weight.detach(),
                k=self.num_concepts,
                n_iter=n_iter,
            )
            self.concept_vectors.copy_(centers)
            self.concept_vectors.add_(torch.randn_like(self.concept_vectors) * 0.01)

            total_positions = self.num_sequences * self.sequence_length
            flat_concept_ids = torch.arange(
                total_positions,
                device=embedding_weight.device,
            )
            flat_concept_ids = flat_concept_ids.remainder(self.num_concepts)
            self.mixture_logits.zero_()
            self.mixture_logits.view(-1, self.num_concepts).scatter_(
                -1,
                flat_concept_ids.unsqueeze(-1),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)


class SequenceConceptSoftTokenDataset(BaseSyntheticTokenDataset):
    """
    Per-sequence concept vectors mixed in embedding space.
    """

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        num_concepts,
        d_model,
        logit_scale=32.0,
        input_mode="probs",
        temperature=1.0,
        init_std=0.02,
    ):
        super().__init__()
        if num_concepts < 1:
            raise ValueError(
                f"Number of concepts must be positive, got {num_concepts}."
            )
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_concepts = num_concepts
        self.d_model = d_model
        self.logit_scale = float(logit_scale)
        if input_mode not in {"probs", "concepts"}:
            raise ValueError(
                "sequence concept input_mode must be either 'probs' or 'concepts', "
                f"got {input_mode!r}."
            )
        self.input_mode = input_mode
        self.temperature = temperature
        self.concept_vectors = nn.Parameter(
            torch.randn(num_sequences, num_concepts, d_model) * init_std
        )
        self.mixture_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, num_concepts) * init_std
        )

    def concept_mixtures(self, indices=None):
        logits = self.mixture_logits if indices is None else self.mixture_logits[indices]
        return torch.softmax(logits / self.temperature, dim=-1)

    def concept_input_embeds(self, indices=None):
        mixtures = self.concept_mixtures(indices)
        concepts = (
            self.concept_vectors
            if indices is None
            else self.concept_vectors[indices]
        )
        return torch.einsum("...tk,...kd->...td", mixtures, concepts)

    def input_embeds(self, indices, embedding_weight):
        if self.input_mode == "concepts":
            return self.concept_input_embeds(indices)
        return self.token_probs(indices, embedding_weight) @ embedding_weight

    def token_probs(self, indices=None, embedding_weight=None):
        if embedding_weight is None:
            raise ValueError("SequenceConceptSoftTokenDataset requires embedding_weight.")
        input_embeds = self.concept_input_embeds(indices)
        logits = input_embeds @ embedding_weight.transpose(0, 1)
        logits = logits * self.logit_scale
        return torch.softmax(logits / self.temperature, dim=-1)

    def initialize_from_token_ids(
        self,
        input_ids,
        confidence=5.0,
        embedding_weight=None,
    ):
        if embedding_weight is None:
            raise ValueError(
                "SequenceConceptSoftTokenDataset initialization requires "
                "embedding_weight."
            )
        if input_ids.shape != (self.num_sequences, self.sequence_length):
            raise ValueError(
                "Expected input_ids shape "
                f"{(self.num_sequences, self.sequence_length)}, got "
                f"{tuple(input_ids.shape)}."
            )
        device = input_ids.device
        source_positions = torch.div(
            torch.arange(self.num_concepts, device=device) * self.sequence_length,
            self.num_concepts,
            rounding_mode="floor",
        ).clamp(max=self.sequence_length - 1)
        concept_tokens = input_ids[:, source_positions]
        position_concept_ids = torch.div(
            torch.arange(self.sequence_length, device=device) * self.num_concepts,
            self.sequence_length,
            rounding_mode="floor",
        ).clamp(max=self.num_concepts - 1)
        with torch.no_grad():
            self.concept_vectors.copy_(embedding_weight[concept_tokens])
            self.concept_vectors.add_(torch.randn_like(self.concept_vectors) * 0.01)

            self.mixture_logits.zero_()
            self.mixture_logits.scatter_(
                -1,
                position_concept_ids.view(1, self.sequence_length, 1).expand(
                    self.num_sequences, -1, -1
                ),
                confidence,
            )
            self.mixture_logits.add_(torch.randn_like(self.mixture_logits) * 0.01)
