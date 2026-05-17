from src.tokenizers.byte_tokenizer import ByteTokenizer
from src.tokenizers.gpt2_bpe_tokenizer import GPT2BPETokenizer
from src.tokenizers.local_bpe_tokenizer import LocalBPETokenizer


def build_tokenizer(tokenizer_type="byte", **kwargs):
    if tokenizer_type in {"byte", "utf8_byte"}:
        return ByteTokenizer()
    if tokenizer_type in {"gpt2", "gpt2_bpe"}:
        return GPT2BPETokenizer(
            tokenizer_name=kwargs.get("tokenizer_name", "gpt2"),
            cache_dir=kwargs.get("cache_dir"),
        )
    if tokenizer_type in {"local_bpe", "bpe"}:
        return LocalBPETokenizer(
            tokenizer_path=kwargs["tokenizer_path"],
            pad_token=kwargs.get("pad_token", "[PAD]"),
            eos_token=kwargs.get("eos_token", "[EOS]"),
            unk_token=kwargs.get("unk_token", "[UNK]"),
        )
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def build_tokenizer_from_dataset_config(dataset_config):
    target = str(dataset_config.get("_target_", ""))

    if target.endswith("TinyStoriesByteDataset"):
        return ByteTokenizer()

    if target.endswith("TinyStoriesBPEDataset"):
        return GPT2BPETokenizer(
            tokenizer_name=dataset_config.get("tokenizer_name", "gpt2"),
            cache_dir=dataset_config.get(
                "tokenizer_cache_dir",
                dataset_config.get("hf_cache_dir"),
            ),
        )

    if target.endswith("TinyStoriesLocalBPEDataset"):
        return LocalBPETokenizer(
            tokenizer_path=dataset_config.get("tokenizer_path"),
            pad_token=dataset_config.get("pad_token", "[PAD]"),
            eos_token=dataset_config.get("eos_token", "[EOS]"),
            unk_token=dataset_config.get("unk_token", "[UNK]"),
        )

    raise ValueError(
        "Could not infer tokenizer from dataset config target: "
        f"{target or '<missing>'}"
    )
