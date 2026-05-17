from pathlib import Path

import hydra
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    input_path = Path(config.get("input_path", "data/tinystories_train.txt"))
    output_dir = Path(
        config.get("output_dir", "artifacts/tokenizers/tinystories_bpe_2048")
    )
    vocab_size = int(config.get("vocab_size", 2048))
    min_frequency = int(config.get("min_frequency", 2))
    pad_token = config.get("pad_token", "[PAD]")
    eos_token = config.get("eos_token", "[EOS]")
    unk_token = config.get("unk_token", "[UNK]")

    if not input_path.exists():
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[pad_token, eos_token, unk_token],
    )
    tokenizer.train([str(input_path)], trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    metadata = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token": pad_token,
        "pad_token_id": tokenizer.token_to_id(pad_token),
        "eos_token": eos_token,
        "eos_token_id": tokenizer.token_to_id(eos_token),
        "unk_token": unk_token,
        "unk_token_id": tokenizer.token_to_id(unk_token),
        "input_path": str(input_path),
    }
    with (output_dir / "metadata.yaml").open("w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(metadata))

    print(OmegaConf.to_yaml(metadata))


if __name__ == "__main__":
    main()
