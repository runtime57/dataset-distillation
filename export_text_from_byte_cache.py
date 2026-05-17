from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.tokenizers import ByteTokenizer


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    input_path = Path(config.get("input_path"))
    output_path = Path(config.get("output_path", "data/exported_from_byte_cache.txt"))
    strip_newlines = bool(config.get("strip_newlines", True))

    if not input_path.exists():
        raise FileNotFoundError(f"Byte-cache file not found: {input_path}")

    token_ids = torch.load(input_path, map_location="cpu")
    if token_ids.dim() != 2:
        raise ValueError(
            "Expected tensor with shape (num_sequences, sequence_length), got "
            f"{tuple(token_ids.shape)}."
        )

    tokenizer = ByteTokenizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in token_ids:
            text = tokenizer.decode(row.tolist())
            if strip_newlines:
                text = text.replace("\n", " ").strip()
            file.write(text)
            file.write("\n")

    print(
        OmegaConf.to_yaml(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "num_sequences": int(token_ids.shape[0]),
                "sequence_length": int(token_ids.shape[1]),
            }
        )
    )


if __name__ == "__main__":
    main()
