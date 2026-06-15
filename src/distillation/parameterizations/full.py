import torch
from torch import nn

from src.distillation.parameterizations.base import BaseSyntheticTokenDataset
from src.utils.io_utils import ROOT_PATH


def _straight_through_hard_probs(probs):
    hard_ids = probs.argmax(dim=-1, keepdim=True)
    hard_probs = torch.zeros_like(probs).scatter_(-1, hard_ids, 1.0)
    return hard_probs.detach() + probs - probs.detach()


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
        hard_forward=False,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.hard_forward = bool(hard_forward)
        self.logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, vocab_size) * init_std
        )

    def token_probs(self, indices=None, embedding_weight=None):
        logits = self.logits if indices is None else self.logits[indices]
        probs = torch.softmax(logits / self.temperature, dim=-1)
        if self.hard_forward:
            probs = _straight_through_hard_probs(probs)
        return probs

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

    def initialize_from_full_logits(
        self,
        full_logits,
        confidence=None,
        embedding_weight=None,
    ):
        if full_logits.shape != (
            self.num_sequences,
            self.sequence_length,
            self.vocab_size,
        ):
            raise ValueError(
                "Expected full_logits shape "
                f"{(self.num_sequences, self.sequence_length, self.vocab_size)}, "
                f"got {tuple(full_logits.shape)}."
            )
        with torch.no_grad():
            self.logits.copy_(
                full_logits.to(device=self.logits.device, dtype=self.logits.dtype)
            )


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
        hard_forward=False,
    ):
        super().__init__()
        if k < 1:
            raise ValueError(f"top-k support size must be positive, got {k}.")
        if k > vocab_size:
            raise ValueError(f"top-k support size {k} exceeds vocab size {vocab_size}.")
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.hard_forward = bool(hard_forward)
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
        probs = probs.scatter_add(-1, support_ids, weights)
        if self.hard_forward:
            probs = _straight_through_hard_probs(probs)
        return probs

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

    def initialize_from_full_logits(
        self,
        full_logits,
        confidence=None,
        embedding_weight=None,
    ):
        if full_logits.shape != (
            self.num_sequences,
            self.sequence_length,
            self.vocab_size,
        ):
            raise ValueError(
                "Expected full_logits shape "
                f"{(self.num_sequences, self.sequence_length, self.vocab_size)}, "
                f"got {tuple(full_logits.shape)}."
            )
        with torch.no_grad():
            full_logits = full_logits.to(
                device=self.logits.device,
                dtype=self.logits.dtype,
            )
            support_ids = full_logits.topk(self.k, dim=-1).indices
            support_logits = full_logits.gather(-1, support_ids)
            if confidence is not None:
                support_logits = support_logits - support_logits.mean(
                    dim=-1,
                    keepdim=True,
                )
                support_logits[..., 0] = float(confidence)
            self.support_ids.copy_(support_ids)
            self.logits.copy_(support_logits)

    def initialize_from_checkpoint(
        self,
        checkpoint,
        confidence=None,
        embedding_weight=None,
    ):
        state_dict = checkpoint.get("synthetic_state_dict", {})
        logits = state_dict.get("logits")
        if (
            torch.is_tensor(logits)
            and logits.dim() == 3
            and logits.shape[-1] == self.vocab_size
        ):
            self.initialize_from_full_logits(
                logits,
                confidence=confidence,
                embedding_weight=embedding_weight,
            )
            return

        target_probs = checkpoint.get("target_probs")
        if torch.is_tensor(target_probs):
            logits = target_probs.clamp_min(1e-8).log()
            self.initialize_from_full_logits(
                logits,
                confidence=confidence,
                embedding_weight=embedding_weight,
            )
            return

        hard_tokens = checkpoint.get("hard_tokens")
        if torch.is_tensor(hard_tokens):
            init_confidence = 5.0 if confidence is None else confidence
            self.initialize_from_token_ids(
                hard_tokens.to(device=self.logits.device),
                confidence=init_confidence,
                embedding_weight=embedding_weight,
            )
            return

        raise KeyError(
            "Checkpoint must contain synthetic_state_dict.logits, target_probs, "
            "or hard_tokens for TopKSoftTokenDataset initialization."
        )

    @torch.no_grad()
    def replace_hard_tokens(
        self,
        sequence_indices,
        position_indices,
        token_ids,
        confidence=None,
        keep_old_hard=True,
        reset_logits=True,
    ):
        """
        Replace the high-probability support token at selected positions.

        The top-k parameterization keeps the real-init token in support slot 0
        with a high logit. This method is the discrete E-step hook: it swaps a
        proposed token into slot 0 and optionally keeps the previous hard token
        in another support slot.
        """
        sequence_indices = sequence_indices.to(
            device=self.support_ids.device,
            dtype=torch.long,
        ).reshape(-1)
        position_indices = position_indices.to(
            device=self.support_ids.device,
            dtype=torch.long,
        ).reshape(-1)
        token_ids = token_ids.to(
            device=self.support_ids.device,
            dtype=torch.long,
        ).reshape(-1)
        if not (
            sequence_indices.numel() == position_indices.numel() == token_ids.numel()
        ):
            raise ValueError(
                "sequence_indices, position_indices, and token_ids must have "
                "the same number of elements."
            )

        changed = 0
        for sequence_index, position_index, token_id in zip(
            sequence_indices.tolist(),
            position_indices.tolist(),
            token_ids.tolist(),
        ):
            support_row = self.support_ids[sequence_index, position_index]
            old_hard = support_row[0].clone()
            if int(old_hard.item()) == int(token_id):
                continue

            existing_slots = (support_row == int(token_id)).nonzero(as_tuple=True)[0]
            if existing_slots.numel() > 0:
                existing_slot = int(existing_slots[0].item())
                support_row[existing_slot] = old_hard
            elif keep_old_hard and self.k > 1:
                support_row[1] = old_hard
            support_row[0] = int(token_id)

            if reset_logits:
                replacement_confidence = (
                    self.logits[sequence_index, position_index].max().detach().clone()
                    if confidence is None
                    else float(confidence)
                )
                self.logits[sequence_index, position_index].zero_()
                self.logits[sequence_index, position_index, 0] = replacement_confidence
            changed += 1
        return changed


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
        gradient_temperature=None,
        init_std=0.02,
        hard_forward=False,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.k = k
        self.temperature = temperature
        self.gradient_temperature = (
            temperature if gradient_temperature is None else gradient_temperature
        )
        self.hard_forward = bool(hard_forward)
        self.logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, vocab_size) * init_std
        )

    def token_probs(self, indices=None, embedding_weight=None):
        logits = self.logits if indices is None else self.logits[indices]
        full_probs = torch.softmax(logits / self.gradient_temperature, dim=-1)

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
        if self.hard_forward:
            sparse_probs = _straight_through_hard_probs(sparse_probs)
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

    def initialize_from_full_logits(
        self,
        full_logits,
        confidence=None,
        embedding_weight=None,
    ):
        if full_logits.shape != (
            self.num_sequences,
            self.sequence_length,
            self.vocab_size,
        ):
            raise ValueError(
                "Expected full_logits shape "
                f"{(self.num_sequences, self.sequence_length, self.vocab_size)}, "
                f"got {tuple(full_logits.shape)}."
            )
        with torch.no_grad():
            full_logits = full_logits.to(
                device=self.logits.device,
                dtype=self.logits.dtype,
            )
            if confidence is not None:
                hard_tokens = full_logits.argmax(dim=-1)
                self.initialize_from_token_ids(
                    hard_tokens,
                    confidence=confidence,
                    embedding_weight=embedding_weight,
                )
                return
            self.logits.copy_(full_logits)

    def initialize_from_checkpoint(
        self,
        checkpoint,
        confidence=None,
        embedding_weight=None,
    ):
        state_dict = checkpoint.get("synthetic_state_dict", {})
        logits = state_dict.get("logits")
        if (
            torch.is_tensor(logits)
            and logits.dim() == 3
            and logits.shape[-1] == self.vocab_size
        ):
            self.initialize_from_full_logits(
                logits,
                confidence=confidence,
                embedding_weight=embedding_weight,
            )
            return

        synthetic_logits = checkpoint.get("synthetic_logits")
        if torch.is_tensor(synthetic_logits):
            self.initialize_from_full_logits(
                synthetic_logits,
                confidence=confidence,
                embedding_weight=embedding_weight,
            )
            return

        target_probs = checkpoint.get("target_probs")
        if torch.is_tensor(target_probs):
            logits = target_probs.clamp_min(1e-8).log()
            self.initialize_from_full_logits(
                logits,
                confidence=confidence,
                embedding_weight=embedding_weight,
            )
            return

        hard_tokens = checkpoint.get("hard_tokens")
        if torch.is_tensor(hard_tokens):
            init_confidence = 5.0 if confidence is None else confidence
            self.initialize_from_token_ids(
                hard_tokens.to(device=self.logits.device),
                confidence=init_confidence,
                embedding_weight=embedding_weight,
            )
            return

        raise KeyError(
            "Checkpoint must contain synthetic_state_dict.logits, "
            "synthetic_logits, target_probs, or hard_tokens for "
            "GumbelTopKSoftTokenDataset initialization."
        )

    @torch.no_grad()
    def replace_hard_tokens(
        self,
        sequence_indices,
        position_indices,
        token_ids,
        confidence=None,
        keep_old_hard=True,
        reset_logits=True,
    ):
        sequence_indices = sequence_indices.to(
            device=self.logits.device,
            dtype=torch.long,
        ).reshape(-1)
        position_indices = position_indices.to(
            device=self.logits.device,
            dtype=torch.long,
        ).reshape(-1)
        token_ids = token_ids.to(
            device=self.logits.device,
            dtype=torch.long,
        ).reshape(-1)
        if not (
            sequence_indices.numel() == position_indices.numel() == token_ids.numel()
        ):
            raise ValueError(
                "sequence_indices, position_indices, and token_ids must have "
                "the same number of elements."
            )

        changed = 0
        for sequence_index, position_index, token_id in zip(
            sequence_indices.tolist(),
            position_indices.tolist(),
            token_ids.tolist(),
        ):
            row = self.logits[sequence_index, position_index]
            old_hard = int(row.argmax().item())
            if old_hard == int(token_id):
                continue

            if reset_logits:
                replacement_confidence = (
                    row.max().detach().clone()
                    if confidence is None
                    else torch.tensor(
                        float(confidence),
                        device=row.device,
                        dtype=row.dtype,
                    )
                )
                old_value = row[old_hard].detach().clone()
                row.zero_()
                row[int(token_id)] = replacement_confidence
                if keep_old_hard:
                    row[old_hard] = min(old_value.item(), 0.0)
            else:
                old_value = row[old_hard].detach().clone()
                row[old_hard] = row[int(token_id)]
                row[int(token_id)] = old_value
            changed += 1
        return changed


def _load_fixed_target_probs(path, expected_shape):
    path = ROOT_PATH / path if not str(path).startswith("/") else path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "target_probs" not in checkpoint:
        raise KeyError(f"Expected target_probs in fixed target checkpoint: {path}")
    target_probs = checkpoint["target_probs"].detach()
    if tuple(target_probs.shape) != tuple(expected_shape):
        raise ValueError(
            "Fixed target checkpoint shape mismatch. "
            f"Expected {tuple(expected_shape)}, got {tuple(target_probs.shape)}."
        )
    return target_probs.contiguous()


class FixedTargetGumbelTopKSoftTokenDataset(GumbelTopKSoftTokenDataset):
    """
    Trainable Gumbel-top-k input tokens with fixed soft target labels.
    """

    uses_decoupled_targets = True

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        k,
        target_checkpoint_path,
        temperature=1.0,
        gradient_temperature=None,
        init_std=0.02,
        hard_forward=False,
    ):
        super().__init__(
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            k=k,
            temperature=temperature,
            gradient_temperature=gradient_temperature,
            init_std=init_std,
            hard_forward=hard_forward,
        )
        if target_checkpoint_path is None:
            raise ValueError(
                "fixed_target_topk_gumbel requires target_checkpoint_path."
            )
        target_probs = _load_fixed_target_probs(
            target_checkpoint_path,
            expected_shape=(num_sequences, sequence_length, vocab_size),
        )
        self.register_buffer("fixed_target_probs", target_probs)

    def target_probs(self, indices=None, embedding_weight=None):
        targets = (
            self.fixed_target_probs
            if indices is None
            else self.fixed_target_probs[indices]
        )
        if embedding_weight is not None:
            targets = targets.to(dtype=embedding_weight.dtype)
        return targets


class DecoupledGumbelTopKSoftTokenDataset(GumbelTopKSoftTokenDataset):
    """
    Trainable Gumbel-top-k input tokens with independently trainable soft labels.
    """

    uses_decoupled_targets = True
    uses_decoupled_inputs = True

    def __init__(
        self,
        num_sequences,
        sequence_length,
        vocab_size,
        k,
        temperature=1.0,
        gradient_temperature=None,
        init_std=0.02,
        target_init_std=None,
        target_init_confidence=None,
        hard_forward=False,
        target_hard_forward=False,
    ):
        super().__init__(
            num_sequences=num_sequences,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            k=k,
            temperature=temperature,
            gradient_temperature=gradient_temperature,
            init_std=init_std,
            hard_forward=hard_forward,
        )
        target_std = init_std if target_init_std is None else target_init_std
        self.target_logits = nn.Parameter(
            torch.randn(num_sequences, sequence_length, vocab_size) * target_std
        )
        self.target_init_confidence = target_init_confidence
        self.target_hard_forward = bool(target_hard_forward)

    def target_probs(self, indices=None, embedding_weight=None):
        logits = self.target_logits if indices is None else self.target_logits[indices]
        probs = torch.softmax(logits / self.temperature, dim=-1)
        if self.target_hard_forward:
            probs = _straight_through_hard_probs(probs)
        return probs

    def input_probs(self, indices=None, embedding_weight=None):
        return self.token_probs(indices=indices, embedding_weight=embedding_weight)

    def initialize_from_token_ids(
        self,
        input_ids,
        confidence=5.0,
        embedding_weight=None,
    ):
        super().initialize_from_token_ids(
            input_ids=input_ids,
            confidence=confidence,
            embedding_weight=embedding_weight,
        )
        target_confidence = (
            confidence
            if self.target_init_confidence is None
            else self.target_init_confidence
        )
        with torch.no_grad():
            self.target_logits.zero_()
            self.target_logits.scatter_(
                -1,
                input_ids.unsqueeze(-1),
                target_confidence,
            )
            self.target_logits.add_(torch.randn_like(self.target_logits) * 0.01)

    def initialize_from_checkpoint(
        self,
        checkpoint,
        confidence=None,
        embedding_weight=None,
    ):
        super().initialize_from_checkpoint(
            checkpoint,
            confidence=confidence,
            embedding_weight=embedding_weight,
        )

        state_dict = checkpoint.get("synthetic_state_dict", {})
        target_logits = state_dict.get("target_logits")
        if torch.is_tensor(target_logits):
            self._copy_target_logits(target_logits)
            return

        target_probs = checkpoint.get("target_probs")
        if torch.is_tensor(target_probs):
            self._copy_target_logits(target_probs.clamp_min(1e-8).log())
            return

        synthetic_logits = checkpoint.get("synthetic_logits")
        if torch.is_tensor(synthetic_logits):
            self._copy_target_logits(synthetic_logits)
            return

        hard_tokens = checkpoint.get("hard_tokens")
        if torch.is_tensor(hard_tokens):
            if self.target_init_confidence is not None:
                target_confidence = self.target_init_confidence
            elif confidence is not None:
                target_confidence = confidence
            else:
                target_confidence = 5.0
            with torch.no_grad():
                self.target_logits.zero_()
                self.target_logits.scatter_(
                    -1,
                    hard_tokens.to(self.target_logits.device).unsqueeze(-1),
                    float(target_confidence),
                )
                self.target_logits.add_(torch.randn_like(self.target_logits) * 0.01)

    def _copy_target_logits(self, target_logits):
        expected_shape = (
            self.num_sequences,
            self.sequence_length,
            self.vocab_size,
        )
        if tuple(target_logits.shape) != expected_shape:
            raise ValueError(
                "Expected target logits shape "
                f"{expected_shape}, got {tuple(target_logits.shape)}."
            )
        with torch.no_grad():
            self.target_logits.copy_(
                target_logits.to(
                    device=self.target_logits.device,
                    dtype=self.target_logits.dtype,
                )
            )
