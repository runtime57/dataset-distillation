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
        use_hard_inputs=False,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.use_hard_inputs = bool(use_hard_inputs)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.checkpoint_config = self._extract_config(checkpoint)
        self.checkpoint_model_config = self._extract_model_config(checkpoint)
        target_probs = self._extract_target_probs(checkpoint, temperature)
        input_probs = self._extract_input_probs(checkpoint, temperature)
        input_ids = self._extract_input_ids(checkpoint)

        if target_probs.dim() != 3:
            raise ValueError(
                "Expected target probabilities with shape "
                "(num_sequences, sequence_length, vocab_size), got "
                f"{tuple(target_probs.shape)}."
            )

        if max_sequences is not None:
            target_probs = target_probs[:max_sequences]
            if input_probs is not None:
                input_probs = input_probs[:max_sequences]
            if input_ids is not None:
                input_ids = input_ids[:max_sequences]

        self.target_probs = target_probs.contiguous()
        self.sequence_length = self.target_probs.shape[1]
        self.vocab_size = self.target_probs.shape[2]
        self.input_probs = None
        if input_probs is not None:
            if tuple(input_probs.shape) != tuple(self.target_probs.shape):
                raise ValueError(
                    "Checkpoint input_probs shape must match target_probs. "
                    f"Expected {tuple(self.target_probs.shape)}, got "
                    f"{tuple(input_probs.shape)}."
                )
            self.input_probs = input_probs.contiguous()
        self.input_ids = None
        if self.use_hard_inputs:
            if input_ids is None:
                raise KeyError(
                    "use_hard_inputs=true requires checkpoint input_ids or "
                    "hard_tokens."
                )
            expected_shape = self.target_probs.shape[:2]
            if tuple(input_ids.shape) != tuple(expected_shape):
                raise ValueError(
                    "Checkpoint input_ids shape must match target_probs "
                    f"sequence shape. Expected {tuple(expected_shape)}, got "
                    f"{tuple(input_ids.shape)}."
                )
            self.input_ids = input_ids.long().contiguous()

    def __len__(self):
        return self.target_probs.shape[0]

    def __getitem__(self, index):
        target_probs = self.target_probs[index].clone()
        item = {
            "target_probs": target_probs,
            "attention_mask": torch.ones(self.sequence_length, dtype=torch.long),
        }
        if self.input_probs is not None and not self.use_hard_inputs:
            item["input_probs"] = self.input_probs[index].clone()
        if self.input_ids is not None:
            input_ids = self.input_ids[index].clone()
            item["input_ids"] = input_ids
            item["labels"] = input_ids.clone()
        return item

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

    @staticmethod
    def _extract_config(checkpoint):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            return checkpoint["config"]
        return None

    @staticmethod
    def _extract_input_ids(checkpoint):
        if not isinstance(checkpoint, dict):
            return None
        for key in ("input_ids", "hard_tokens", "token_ids"):
            value = checkpoint.get(key)
            if torch.is_tensor(value):
                return value.long()
        return None

    @classmethod
    def _extract_input_probs(cls, checkpoint, temperature):
        if not isinstance(checkpoint, dict):
            return None
        input_probs = checkpoint.get("input_probs")
        if torch.is_tensor(input_probs):
            return cls._normalize_probs(input_probs.float(), temperature)
        for key in ("synthetic_input_logits", "input_logits"):
            value = checkpoint.get(key)
            if torch.is_tensor(value):
                return torch.softmax(value.float() / temperature, dim=-1)
        return None

    @classmethod
    def _extract_model_config(cls, checkpoint):
        config = cls._extract_config(checkpoint)
        if isinstance(config, dict) and isinstance(config.get("model"), dict):
            return config["model"]
        return None

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
