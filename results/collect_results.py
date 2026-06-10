from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"

RUN_NAMES = [
    "results_real_fixed15k_seq256_lrsched_verify",
    "results_softdistill_anchors1024_big_verify",
    "results_softdistill_concepts64_scale32_verify",
    "results_softdistill_seqanchors256_verify",
]

EPOCH_RE = re.compile(r"epoch\s+:\s+(\d+)")
PPL_RE = {
    "train_ppl": re.compile(r"- train - INFO -\s+Perplexity\s+:\s+([0-9.eE+-]+)$"),
    "val_ppl": re.compile(r"val_Perplexity\s+:\s+([0-9.eE+-]+)"),
    "test_ppl": re.compile(r"test_Perplexity:\s+([0-9.eE+-]+)"),
}


def parse_log(log_path: Path) -> dict:
    current: dict[str, float | int] = {}
    rows: list[dict] = []
    for line in log_path.read_text().splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            if {"epoch", "train_ppl", "val_ppl", "test_ppl"} <= current.keys():
                rows.append(current.copy())
            current = {"epoch": int(epoch_match.group(1))}
            continue
        for key, pattern in PPL_RE.items():
            match = pattern.search(line)
            if match:
                current[key] = float(match.group(1))
    if {"epoch", "train_ppl", "val_ppl", "test_ppl"} <= current.keys():
        rows.append(current.copy())
    if not rows:
        raise RuntimeError(f"No epoch rows found in {log_path}")
    best = min(rows, key=lambda row: row["val_ppl"])
    return {
        "best_epoch": best["epoch"],
        "train_ppl": best["train_ppl"],
        "val_ppl": best["val_ppl"],
        "test_ppl": best["test_ppl"],
    }


def main() -> None:
    out: dict[str, dict] = {}
    for run_name in RUN_NAMES:
        log_path = RUNS_DIR / run_name / "info.log"
        if not log_path.exists():
            continue
        try:
            out[run_name] = parse_log(log_path)
        except RuntimeError:
            continue
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
