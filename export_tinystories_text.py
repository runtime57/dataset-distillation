from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    dataset_name = config.get("dataset_name", "roneneldan/TinyStories")
    split = config.get("split", "train")
    text_field = config.get("text_field", "text")
    max_texts = int(config.get("max_texts", 2000))
    streaming = bool(config.get("streaming", True))
    cache_dir = config.get("hf_cache_dir")
    output_path = Path(config.get("output_path", f"data/{split}.txt"))

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            text = row.get(text_field)
            if not text:
                continue
            file.write(text.replace("\n", " ").strip())
            file.write("\n")
            written += 1
            if written >= max_texts:
                break

    print(
        OmegaConf.to_yaml(
            {
                "output_path": str(output_path),
                "written_texts": written,
                "split": split,
                "dataset_name": dataset_name,
            }
        )
    )


if __name__ == "__main__":
    main()
