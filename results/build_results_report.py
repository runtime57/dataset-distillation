from __future__ import annotations

import json
from pathlib import Path

import yaml

from collect_results import parse_log


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
REPORT_PATH = RESULTS_DIR / "RESULTS.md"

RUNS = [
    {
        "run_name": "results_real_fixed15k_seq256_lrsched_verify",
        "training_type": "real",
        "memoization_type": "none",
        "upstream_dir": None,
    },
    {
        "run_name": "results_softdistill_anchors1024_big_verify",
        "training_type": "softdistill",
        "memoization_type": "anchors",
        "upstream_dir": ROOT
        / "saved/distillation/distill256_local_bpe_1024_gm_anchors1024_ent1e3_continue_from100",
    },
    {
        "run_name": "results_softdistill_concepts64_scale32_verify",
        "training_type": "softdistill",
        "memoization_type": "concepts",
        "upstream_dir": ROOT
        / "saved/distillation/distill256_local_bpe_1024_gm_concepts64_debug_concepts_scale32_lr1e3",
    },
    {
        "run_name": "results_softdistill_seqanchors256_verify",
        "training_type": "softdistill",
        "memoization_type": "sequence_anchors",
        "upstream_dir": ROOT
        / "saved/distillation/distill256_local_bpe_1024_gm_seqanchors256_n256_ilr1e2",
    },
]

BASE_COLUMNS = [
    "run",
    "training_type",
    "train_dataset_type",
    "train_dataset_size",
    "val_dataset_size",
    "test_dataset_size",
    "train.tokenizer_path",
    "train.local_text_path",
    "train.checkpoint_path",
    "val.local_text_path",
    "test.local_text_path",
    "memoization_type",
    "k",
]

MODEL_COLUMNS = [
    "model.vocab_size",
    "model.max_seq_len",
    "model.d_model",
    "model.n_heads",
    "model.n_layers",
    "model.dim_feedforward",
    "model.dropout",
    "model.pad_token_id",
]

TRAIN_COLUMNS = [
    "dataloader.batch_size",
    "dataloader.num_workers",
    "dataloader.pin_memory",
    "optimizer",
    "optimizer.lr",
    "optimizer.weight_decay",
    "lr_scheduler",
    "lr_scheduler.gamma",
    "lr_scheduler.step_size",
    "trainer.n_epochs",
    "trainer.epoch_len",
    "trainer.device",
    "trainer.seed",
    "trainer.max_grad_norm",
]

SYNTH_COLUMNS = [
    "synthetic.num_sequences",
    "synthetic.sequence_length",
    "synthetic.batch_size",
    "synthetic.temperature",
    "synthetic.init_std",
    "synthetic.init_from_real",
    "synthetic.init_confidence",
    "synthetic.parameterization",
    "synthetic.num_anchors",
    "synthetic.num_concepts",
    "synthetic.concept_input_mode",
    "synthetic.concept_logit_scale",
    "synthetic.alternating.enabled",
    "synthetic.alternating.mode",
    "synthetic.alternating.mixture_steps",
    "synthetic.alternating.component_steps",
]

DISTILL_COLUMNS = [
    "distill_objective",
    "distill_inner_lr",
    "distill_outer_batches",
    "distill_entropy_weight",
    "distill_n_steps",
    "distill_log_step",
    "distill_save_period",
    "distill_best_inner_loss",
]

RESULT_COLUMNS = [
    "best_epoch",
    "train_ppl",
    "val_ppl",
    "test_ppl",
    "config",
    "artifact_dir",
    "distill_config",
    "decoded_samples",
]


def flatten_model(config: dict) -> dict:
    out = {}
    for key, value in config["model"].items():
        if key == "_target_":
            continue
        out[f"model.{key}"] = value
    return out


def nested_get(mapping: dict, *keys, default="-"):
    current = mapping
    for key in keys:
        if current is None or key not in current:
            return default
        current = current[key]
    return current


def resolve_value(config: dict, value):
    if not isinstance(value, str):
        return value
    if not (value.startswith("${") and value.endswith("}")):
        return value
    path = value[2:-1].split(".")
    current = config
    for key in path:
        current = current[key]
    return current


def best_inner_loss(upstream_dir: Path | None):
    if upstream_dir is None:
        return "-"
    metrics_path = upstream_dir / "distill_metrics.jsonl"
    if not metrics_path.exists():
        return "-"
    best = None
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        value = row.get("inner_loss")
        if value is None:
            continue
        best = value if best is None else min(best, value)
    return "-" if best is None else f"{best:.6f}"


def md_link(path: Path | None, label: str | None = None) -> str:
    if path is None:
        return "-"
    resolved = path.resolve()
    return f"[{label or path.name}]({resolved})"


def detect_train_size(config: dict) -> tuple[str, str, str, str]:
    train_cfg = config["datasets"]["train"]
    if "checkpoint_path" in train_cfg:
        return (
            "synthetic_checkpoint",
            str(nested_get(train_cfg, "max_sequences", default="all")),
            str(config["datasets"]["val"]["max_texts"]),
            str(config["datasets"]["test"]["max_texts"]),
        )
    return (
        "real_text",
        str(train_cfg["max_texts"]),
        str(config["datasets"]["val"]["max_texts"]),
        str(config["datasets"]["test"]["max_texts"]),
    )


def row_for_run(spec: dict) -> dict:
    run_dir = RUNS_DIR / spec["run_name"]
    config_path = run_dir / "config.yaml"
    log_path = run_dir / "info.log"
    config = yaml.safe_load(config_path.read_text())
    metrics = parse_log(log_path)
    row = {
        "run": spec["run_name"],
        "training_type": spec["training_type"],
        "memoization_type": spec["memoization_type"],
        "config": md_link(config_path, "config.yaml"),
        "artifact_dir": md_link(run_dir, spec["run_name"]),
        "distill_config": "-",
        "decoded_samples": "-",
    }
    row.update(flatten_model(config))
    train_cfg = config["datasets"]["train"]
    train_dataset_type, train_dataset_size, val_dataset_size, test_dataset_size = detect_train_size(config)
    row["train_dataset_type"] = train_dataset_type
    row["train_dataset_size"] = train_dataset_size
    row["val_dataset_size"] = val_dataset_size
    row["test_dataset_size"] = test_dataset_size
    row["train.tokenizer_path"] = train_cfg.get("tokenizer_path", "-")
    row["train.local_text_path"] = train_cfg.get("local_text_path", "-")
    row["train.checkpoint_path"] = train_cfg.get("checkpoint_path", "-")
    row["val.local_text_path"] = config["datasets"]["val"].get("local_text_path", "-")
    row["test.local_text_path"] = config["datasets"]["test"].get("local_text_path", "-")
    row["dataloader.batch_size"] = resolve_value(config, config["dataloader"]["batch_size"])
    row["dataloader.num_workers"] = resolve_value(config, config["dataloader"]["num_workers"])
    row["dataloader.pin_memory"] = resolve_value(config, config["dataloader"]["pin_memory"])
    row["optimizer"] = config["optimizer"]["_target_"].split(".")[-1]
    row["optimizer.lr"] = config["optimizer"]["lr"]
    row["optimizer.weight_decay"] = config["optimizer"]["weight_decay"]
    row["lr_scheduler"] = config["lr_scheduler"]["_target_"].split(".")[-1]
    row["lr_scheduler.gamma"] = resolve_value(config, config["lr_scheduler"]["gamma"])
    row["lr_scheduler.step_size"] = resolve_value(config, config["lr_scheduler"]["step_size"])
    row["trainer.n_epochs"] = resolve_value(config, config["trainer"]["n_epochs"])
    row["trainer.epoch_len"] = resolve_value(config, config["trainer"]["epoch_len"])
    row["trainer.device"] = resolve_value(config, config["trainer"]["device"])
    row["trainer.seed"] = resolve_value(config, config["trainer"]["seed"])
    row["trainer.max_grad_norm"] = resolve_value(config, config["trainer"]["max_grad_norm"])
    row.update(metrics)

    checkpoint_path = train_cfg.get("checkpoint_path")
    if checkpoint_path:
        upstream_cfg = yaml.safe_load(spec["upstream_dir"].joinpath("config.yaml").read_text())
        synthetic = upstream_cfg["distillation"]["synthetic"]
        row["synthetic.num_sequences"] = resolve_value(upstream_cfg, synthetic.get("num_sequences", "-"))
        row["synthetic.sequence_length"] = resolve_value(upstream_cfg, synthetic.get("sequence_length", "-"))
        row["synthetic.batch_size"] = resolve_value(upstream_cfg, synthetic.get("batch_size", "-"))
        row["synthetic.temperature"] = resolve_value(upstream_cfg, synthetic.get("temperature", "-"))
        row["synthetic.init_std"] = resolve_value(upstream_cfg, synthetic.get("init_std", "-"))
        row["synthetic.init_from_real"] = resolve_value(upstream_cfg, synthetic.get("init_from_real", "-"))
        row["synthetic.init_confidence"] = resolve_value(upstream_cfg, synthetic.get("init_confidence", "-"))
        row["synthetic.parameterization"] = resolve_value(upstream_cfg, synthetic.get("parameterization", "-"))
        row["synthetic.num_anchors"] = resolve_value(upstream_cfg, synthetic.get("num_anchors", "-"))
        row["synthetic.num_concepts"] = resolve_value(upstream_cfg, synthetic.get("num_concepts", "-"))
        row["synthetic.concept_input_mode"] = resolve_value(upstream_cfg, synthetic.get("concept_input_mode", "-"))
        row["synthetic.concept_logit_scale"] = resolve_value(upstream_cfg, synthetic.get("concept_logit_scale", "-"))
        alternating = synthetic.get("alternating", {})
        row["synthetic.alternating.enabled"] = alternating.get("enabled", "-")
        row["synthetic.alternating.mode"] = alternating.get("mode", "-")
        row["synthetic.alternating.mixture_steps"] = alternating.get("mixture_steps", "-")
        row["synthetic.alternating.component_steps"] = alternating.get("component_steps", "-")
        row["k"] = synthetic.get("num_anchors", synthetic.get("num_concepts", "-"))
        row["distill_objective"] = upstream_cfg["distillation"]["objective"]
        row["distill_inner_lr"] = upstream_cfg["distillation"]["inner_lr"]
        row["distill_outer_batches"] = upstream_cfg["distillation"]["outer_batches"]
        row["distill_entropy_weight"] = upstream_cfg["distillation"]["entropy_weight"]
        row["distill_n_steps"] = upstream_cfg["distillation"]["n_steps"]
        row["distill_log_step"] = upstream_cfg["distillation"]["log_step"]
        row["distill_save_period"] = upstream_cfg["distillation"]["save_period"]
        row["distill_best_inner_loss"] = best_inner_loss(spec["upstream_dir"])
        local_upstream_dir = run_dir / "upstream"
        local_distill_config = local_upstream_dir / "distill_config.yaml"
        row["distill_config"] = md_link(
            local_distill_config if local_distill_config.exists() else spec["upstream_dir"] / "config.yaml",
            "distill config",
        )
        decoded = local_upstream_dir / "decoded_samples_best.txt"
        if not decoded.exists():
            decoded = local_upstream_dir / "decoded_samples.txt"
        if not decoded.exists():
            decoded = spec["upstream_dir"] / "decoded_samples_best.txt"
        if not decoded.exists():
            decoded = spec["upstream_dir"] / "decoded_samples.txt"
        row["decoded_samples"] = md_link(decoded, decoded.name) if decoded.exists() else "-"
        row["train_dataset_size"] = row["synthetic.num_sequences"]
    else:
        for column in SYNTH_COLUMNS + DISTILL_COLUMNS:
            row[column] = "-"
        row["k"] = "-"

    return row


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "-")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    rows = []
    for spec in RUNS:
        log_path = RUNS_DIR / spec["run_name"] / "info.log"
        if not log_path.exists():
            continue
        try:
            rows.append(row_for_run(spec))
        except RuntimeError:
            continue
    columns = BASE_COLUMNS + MODEL_COLUMNS + TRAIN_COLUMNS + SYNTH_COLUMNS + DISTILL_COLUMNS + RESULT_COLUMNS
    lines = [
        "# Reproduced Results",
        "",
        "Fresh reruns collected on branch `results`. The table flattens the saved configs and keeps all model fields except the `_target_`/`src` plumbing.",
        "",
        markdown_table(rows, columns),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
