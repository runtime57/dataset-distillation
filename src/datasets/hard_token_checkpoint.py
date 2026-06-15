from pathlib import Path

import torch
from torch.utils.data import Dataset


class HardTokenCheckpointDataset(Dataset):
    """
    Dataset backed by hard token ids stored in a distilled checkpoint.
    """

    def __init__(self, checkpoint_path, max_sequences=None):
        self.checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.checkpoint_config = self._extract_config(checkpoint)
        self.checkpoint_model_config = self._extract_model_config(checkpoint)
        token_ids = self._extract_token_ids(checkpoint)

        if token_ids.dim() != 2:
            raise ValueError(
                "Expected hard tokens with shape "
                f"(num_sequences, sequence_length), got {tuple(token_ids.shape)}."
            )

        if max_sequences is not None:
            token_ids = token_ids[:max_sequences]

        self.input_ids = token_ids.long()
        self.sequence_length = self.input_ids.shape[1]
        self.vocab_size = (
            None
            if self.checkpoint_model_config is None
            else self.checkpoint_model_config.get("vocab_size")
        )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        input_ids = self.input_ids[index].clone()
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones(self.sequence_length, dtype=torch.long),
        }

    @staticmethod
    def _extract_token_ids(checkpoint):
        if torch.is_tensor(checkpoint):
            return checkpoint
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "HardTokenCheckpointDataset expects a tensor checkpoint or a dict "
                "containing hard token ids."
            )

        for key in ("hard_tokens", "input_ids", "token_ids"):
            value = checkpoint.get(key)
            if torch.is_tensor(value):
                return value

        raise KeyError(
            "Could not find hard token ids in checkpoint. "
            "Expected one of: hard_tokens, input_ids, token_ids."
        )

    @staticmethod
    def _extract_config(checkpoint):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            return checkpoint["config"]
        return None

    @classmethod
    def _extract_model_config(cls, checkpoint):
        config = cls._extract_config(checkpoint)
        if isinstance(config, dict) and isinstance(config.get("model"), dict):
            return config["model"]
        return None
