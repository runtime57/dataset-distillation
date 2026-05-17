from pathlib import Path

import torch
from torch.utils.data import Dataset


class SoftTokenCheckpointDataset(Dataset):
    """
    Dataset backed by a distilled soft-token checkpoint.
    """

    def __init__(
        self,
        checkpoint_path,
        temperature=1.0,
        max_sequences=None,
    ):
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        target_probs = self._extract_target_probs(checkpoint, temperature)

        if target_probs.dim() != 3:
            raise ValueError(
                "Expected target probabilities with shape "
                "(num_sequences, sequence_length, vocab_size), got "
                f"{tuple(target_probs.shape)}."
            )

        if max_sequences is not None:
            target_probs = target_probs[:max_sequences]

        self.target_probs = target_probs.contiguous()
        self.sequence_length = self.target_probs.shape[1]

    def __len__(self):
        return self.target_probs.shape[0]

    def __getitem__(self, index):
        target_probs = self.target_probs[index].clone()
        return {
            "target_probs": target_probs,
            "attention_mask": torch.ones(self.sequence_length, dtype=torch.long),
        }

    @staticmethod
    def _extract_logits(checkpoint):
        if torch.is_tensor(checkpoint):
            return checkpoint.float()
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "SoftTokenCheckpointDataset expects a tensor checkpoint or a dict "
                "containing synthetic logits."
            )

        for key in ("synthetic_logits", "logits"):
            value = checkpoint.get(key)
            if torch.is_tensor(value):
                return value.float()

        raise KeyError(
            "Could not find synthetic logits in checkpoint. "
            "Expected one of: synthetic_logits, logits."
        )

    @classmethod
    def _extract_target_probs(cls, checkpoint, temperature):
        if isinstance(checkpoint, dict):
            target_probs = checkpoint.get("target_probs")
            if torch.is_tensor(target_probs):
                return cls._normalize_probs(target_probs.float(), temperature)

        logits = cls._extract_logits(checkpoint)
        return torch.softmax(logits / temperature, dim=-1)

    @staticmethod
    def _normalize_probs(probs, temperature):
        if temperature != 1.0:
            logits = probs.clamp_min(1e-8).log()
            return torch.softmax(logits / temperature, dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
