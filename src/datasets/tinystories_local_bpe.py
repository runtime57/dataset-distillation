import hashlib
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.tokenizers import LocalBPETokenizer
from src.utils.io_utils import ROOT_PATH


class TinyStoriesLocalBPEDataset(Dataset):
    """
    TinyStories dataset backed by a locally trained BPE tokenizer.
    """

    def __init__(
        self,
        split="train",
        dataset_name="roneneldan/TinyStories",
        text_field="text",
        tokenizer_path="artifacts/tokenizers/tinystories_bpe_2048/tokenizer.json",
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        sequence_length=128,
        max_texts=2000,
        max_sequences=1024,
        streaming=True,
        hf_cache_dir=None,
        data_dir="data/tinystories_local_bpe",
        local_text_path=None,
        add_eos=True,
        use_cache=True,
    ):
        self.sequence_length = sequence_length
        self.tokenizer = LocalBPETokenizer(
            tokenizer_path=tokenizer_path,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )

        cache_path = self._cache_path(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            tokenizer_path=tokenizer_path,
            sequence_length=sequence_length,
            max_texts=max_texts,
            max_sequences=max_sequences,
            local_text_path=local_text_path,
        )

        if use_cache and cache_path.exists():
            self.input_ids = torch.load(cache_path)
            return

        texts = self._iter_texts(
            split=split,
            dataset_name=dataset_name,
            text_field=text_field,
            max_texts=max_texts,
            streaming=streaming,
            hf_cache_dir=hf_cache_dir,
            local_text_path=local_text_path,
        )
        self.input_ids = self._build_sequences(
            texts=texts,
            sequence_length=sequence_length,
            max_sequences=max_sequences,
            add_eos=add_eos,
        )

        if use_cache:
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.input_ids, cache_path)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        input_ids = self.input_ids[index].clone()
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }

    def _build_sequences(self, texts, sequence_length, max_sequences, add_eos):
        token_ids = []
        chunks = []

        for text in texts:
            token_ids.extend(self.tokenizer.encode(text, add_eos=add_eos))

            while len(token_ids) >= sequence_length:
                chunks.append(token_ids[:sequence_length])
                token_ids = token_ids[sequence_length:]
                if max_sequences is not None and len(chunks) >= max_sequences:
                    return torch.tensor(chunks, dtype=torch.long)

        if not chunks:
            raise ValueError(
                "No full token sequences were produced. Increase max_texts or "
                "decrease sequence_length."
            )
        return torch.tensor(chunks, dtype=torch.long)

    def _iter_texts(
        self,
        split,
        dataset_name,
        text_field,
        max_texts,
        streaming,
        hf_cache_dir,
        local_text_path,
    ):
        if local_text_path is not None:
            text_path = Path(local_text_path)
            with text_path.open("r", encoding="utf-8") as file:
                lines = (line.strip() for line in file)
                lines = (line for line in lines if line)
                rows = lines if max_texts is None else islice(lines, max_texts)
                yield from rows
            return

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "TinyStoriesLocalBPEDataset requires the 'datasets' package. "
                "Install dependencies with 'pip install -r requirements.txt'."
            ) from exc

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=hf_cache_dir,
        )

        if not streaming and max_texts is not None:
            dataset = dataset.select(range(min(max_texts, len(dataset))))

        rows = dataset if max_texts is None else islice(dataset, max_texts)
        for row in rows:
            text = row[text_field]
            if text:
                yield text

    @staticmethod
    def _cache_path(
        data_dir,
        dataset_name,
        split,
        tokenizer_path,
        sequence_length,
        max_texts,
        max_sequences,
        local_text_path,
    ):
        dataset_slug = TinyStoriesLocalBPEDataset._slug(dataset_name)
        split_slug = TinyStoriesLocalBPEDataset._slug(split)
        tokenizer_hash = hashlib.sha1(str(tokenizer_path).encode("utf-8")).hexdigest()[:8]
        source_slug = f"{dataset_slug}_localbpe_{tokenizer_hash}"
        if local_text_path is not None:
            path = Path(local_text_path).expanduser()
            try:
                path = path.resolve()
            except OSError:
                pass
            path_hash = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
            source_slug = (
                f"local_{TinyStoriesLocalBPEDataset._slug(path.stem)}_"
                f"{path_hash}_localbpe_{tokenizer_hash}"
            )
        filename = (
            f"{source_slug}_{split_slug}_seq{sequence_length}"
            f"_texts{max_texts}_chunks{max_sequences}.pt"
        )
        return ROOT_PATH / data_dir / filename

    @staticmethod
    def _slug(value):
        return "".join(char if char.isalnum() else "_" for char in str(value))
