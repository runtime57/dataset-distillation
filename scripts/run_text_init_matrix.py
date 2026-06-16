#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SAVE_DIR = "saved/text_matrix_20260616"
MATRIX_EVAL_DIR = "saved/text_matrix_20260616_eval"
RESULTS_JSON = ROOT / "results_text_matrix_2026-06-16.json"
RESULTS_MD = ROOT / "results_text_matrix_2026-06-16.md"
LOG_DIR = ROOT / "logs/text_matrix_20260616"
EXPERT_SP4_DIR = (
    "saved/expert_trajectories/"
    "tiny_lm_local_bpe_1024_seq256_adamw_lr35e4_wd11_sp4"
)
EXPERT_SP4_PATH = ROOT / EXPERT_SP4_DIR / "expert_checkpoints.pth"


OBJECTIVES = ("gm", "fm", "tm", "os")
INITS = ("real", "noise", "real-many", "real-many-4")
PARAMS = ("topk", "argmax")

BASE_CONFIGS = {
    "gm": "distill_soft_tokens_gm_conditional_random_gumbel64_n256_local_bpe_1024",
    "fm": "distill_soft_tokens_fm_conditional_random_gumbel64_n256_local_bpe_1024",
    "tm": "distill_soft_tokens_tm_adamw_gumbel64_n256_local_bpe_1024",
    "os": "distill_soft_tokens_os_gumbel64_matched_local_bpe_1024",
}

EXISTING_RUNS = {
    ("gm", "noise", "topk"): (
        "saved/conditional_text/distill_gm_conditional_random_gumbel64_n256_g16_s200",
        "saved/conditional_text_eval/eval_condgm_random_gumbel64_n256_g16_s200_best_softinput",
    ),
    ("gm", "real", "topk"): (
        "saved/conditional_text/distill_gm_conditional_realinit_gumbel64_n256_g16_s200",
        "saved/conditional_text_eval/eval_condgm_realinit1_gumbel64_n256_g16_s200_best_softinput",
    ),
    ("gm", "real-many", "topk"): (
        "saved/conditional_text/distill_gm_conditional_realmix_gumbel64_n256_g16_s200",
        "saved/conditional_text_eval/eval_condgm_realmix_gumbel64_n256_g16_s200_best_softinput",
    ),
}


@dataclass(frozen=True)
class Combo:
    objective: str
    init: str
    param: str

    @property
    def init_id(self):
        return self.init.replace("-", "")

    @property
    def run_name(self):
        return (
            f"textmatrix_{self.objective}_{self.init_id}_{self.param}"
            "_gumbel64_n256_s200"
        )

    @property
    def eval_name(self):
        return f"eval_{self.run_name}_best_softinput"

    @property
    def key(self):
        return (self.objective, self.init, self.param)


def repo_path(relative_path):
    return ROOT / relative_path


def distill_dir(combo):
    existing = EXISTING_RUNS.get(combo.key)
    if existing is not None:
        return repo_path(existing[0])
    return ROOT / MATRIX_SAVE_DIR / combo.run_name


def eval_dir(combo):
    existing = EXISTING_RUNS.get(combo.key)
    if existing is not None:
        return repo_path(existing[1])
    return ROOT / MATRIX_EVAL_DIR / combo.eval_name


def run_logged(command, log_path, dry_run=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def ensure_expert_trajectory(dry_run=False):
    if EXPERT_SP4_PATH.exists():
        return
    command = [
        "python3",
        "-m",
        "src.compute_expert_trajectory",
        "--config-name",
        "expert_trajectory_seq256_adamw_local_bpe_1024",
        "expert.n_steps=256",
        "expert.save_period=4",
        f"expert.save_dir={EXPERT_SP4_DIR}",
    ]
    run_logged(command, LOG_DIR / "expert_sp4.log", dry_run=dry_run)


def init_overrides(combo):
    if combo.init == "real":
        overrides = [
            "distillation.synthetic.init_from_real=true",
            "++distillation.synthetic.init_mode=real",
        ]
        if combo.objective in ("gm", "fm"):
            overrides.append(
                "distillation.conditional_matching.synthetic_group_method=init_prefix_hash"
            )
        return overrides

    if combo.init == "noise":
        overrides = [
            "distillation.synthetic.init_from_real=false",
            "++distillation.synthetic.init_mode=real",
        ]
        if combo.objective in ("gm", "fm"):
            overrides.append(
                "distillation.conditional_matching.synthetic_group_method=balanced"
            )
        return overrides

    if combo.init in ("real-many", "real-many-4"):
        max_sequences = "1024" if combo.init == "real-many-4" else "null"
        overrides = [
            "distillation.synthetic.init_from_real=false",
            "++distillation.synthetic.init_mode=real_mixture",
            "++distillation.synthetic.init_mixture_eps=1e-4",
            "++distillation.synthetic.init_mixture_offset=0",
            f"++distillation.synthetic.init_mixture_max_sequences={max_sequences}",
        ]
        if combo.objective in ("gm", "fm"):
            overrides.append(
                "distillation.conditional_matching.synthetic_group_method=balanced"
            )
        return overrides

    raise ValueError(f"Unknown init: {combo.init}")


def distill_command(combo):
    hard_forward = "true" if combo.param == "argmax" else "false"
    command = [
        "python3",
        "distill.py",
        "--config-name",
        BASE_CONFIGS[combo.objective],
        f"distillation.run_name={combo.run_name}",
        f"distillation.save_dir={MATRIX_SAVE_DIR}",
        "distillation.override=true",
        "distillation.n_steps=200",
        "distillation.log_step=20",
        "distillation.save_period=200",
        "distillation.save_step_checkpoints=false",
        "distillation.synthetic.parameterization=topk_gumbel",
        "distillation.synthetic.topk=64",
        "++distillation.synthetic.gradient_temperature=2.0",
        f"++distillation.synthetic.hard_forward={hard_forward}",
    ]
    command.extend(init_overrides(combo))
    if combo.objective == "tm":
        command.extend(
            [
                "distillation.outer_batches=1",
                "distillation.n_inner_steps=4",
                f"distillation.expert_trajectory_path={EXPERT_SP4_DIR}/expert_checkpoints.pth",
            ]
        )
    return command


def eval_command(combo):
    checkpoint = distill_dir(combo) / "full_soft_tokens_best.pth"
    return [
        "python3",
        "train.py",
        "--config-name",
        "tinystories_lm_softdistill_matched_local_bpe_1024",
        f"datasets.train.checkpoint_path={checkpoint.relative_to(ROOT)}",
        f"writer.run_name={combo.eval_name}",
        f"trainer.save_dir={MATRIX_EVAL_DIR}",
    ]


def parse_distill(combo):
    metrics_path = distill_dir(combo) / "distill_metrics.jsonl"
    if not metrics_path.exists():
        return None
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None
    best = min(rows, key=lambda row: row.get("inner_loss", float("inf")))
    first = rows[0]
    last = rows[-1]
    return {
        "steps": len(rows),
        "start_inner": first.get("inner_loss"),
        "best_inner": best.get("inner_loss"),
        "best_inner_step": best.get("step"),
        "final_inner": last.get("inner_loss"),
        "final_entropy": last.get("entropy"),
    }


def parse_eval(combo):
    info_path = eval_dir(combo) / "info.log"
    if not info_path.exists():
        return None
    text = info_path.read_text(encoding="utf-8")
    chunks = text.split("    epoch          : ")[1:]
    rows = []
    for chunk in chunks:
        epoch = int(chunk.splitlines()[0].strip())

        def value(key):
            match = re.search(rf"{re.escape(key)}\s*:\s*([0-9.eE+-]+)", chunk)
            return None if match is None else float(match.group(1))

        rows.append(
            {
                "epoch": epoch,
                "train_ppl": value("Perplexity"),
                "val_ppl": value("val_Perplexity"),
                "test_ppl": value("test_Perplexity"),
            }
        )
    if not rows:
        return None
    return min(rows, key=lambda row: row["val_ppl"])


def cleanup_weights(combo):
    for directory in (distill_dir(combo), eval_dir(combo)):
        if not directory.exists():
            continue
        for path in directory.glob("*.pth"):
            path.unlink()


def collect_results(combos):
    results = []
    for combo in combos:
        results.append(
            {
                "objective": combo.objective,
                "init": combo.init,
                "param": combo.param,
                "distill_run": distill_dir(combo).relative_to(ROOT).as_posix(),
                "eval_run": eval_dir(combo).relative_to(ROOT).as_posix(),
                "distill": parse_distill(combo),
                "eval": parse_eval(combo),
            }
        )
    return results


def load_existing_results():
    if not RESULTS_JSON.exists():
        return []
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


def merge_results(existing, updates):
    merged = {
        (row["objective"], row["init"], row["param"]): row
        for row in existing
    }
    for row in updates:
        merged[(row["objective"], row["init"], row["param"])] = row
    seen = set()
    ordered = []
    for row in existing:
        key = (row["objective"], row["init"], row["param"])
        if key in merged and key not in seen:
            ordered.append(merged[key])
            seen.add(key)
    for row in updates:
        key = (row["objective"], row["init"], row["param"])
        if key not in seen:
            ordered.append(row)
            seen.add(key)
    return ordered


def write_results(results):
    RESULTS_JSON.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Text Init Matrix, 2026-06-16",
        "",
        "Canonical setup: TinyTransformerLM 4-layer, local BPE-1024, "
        "`n=256`, `max_seq_len=256`. `topk` means topk-gumbel64 "
        "soft-forward; `argmax` means the same parameterization with "
        "`hard_forward=true`.",
        "",
        "`real-many` uses all 10k train sequences, about 39-40 source "
        "texts per synthetic sequence. `real-many-4` caps the mixture at "
        "1024 source sequences, i.e. about 4 source texts per synthetic "
        "sequence.",
        "",
        "| objective | init | param | val PPL | test PPL | epoch | best inner | entropy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        eval_row = row["eval"] or {}
        distill_row = row["distill"] or {}
        lines.append(
            "| {objective} | {init} | {param} | {val} | {test} | {epoch} | {inner} | {entropy} |".format(
                objective=row["objective"],
                init=row["init"],
                param=row["param"],
                val=(
                    "pending"
                    if eval_row.get("val_ppl") is None
                    else f"{eval_row['val_ppl']:.3f}"
                ),
                test=(
                    "pending"
                    if eval_row.get("test_ppl") is None
                    else f"{eval_row['test_ppl']:.3f}"
                ),
                epoch=(
                    "pending"
                    if eval_row.get("epoch") is None
                    else str(eval_row["epoch"])
                ),
                inner=(
                    "pending"
                    if distill_row.get("best_inner") is None
                    else f"{distill_row['best_inner']:.6f}"
                ),
                entropy=(
                    "pending"
                    if distill_row.get("final_entropy") is None
                    else f"{distill_row['final_entropy']:.3f}"
                ),
            )
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list(raw, allowed):
    if raw == "all":
        return allowed
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown values {unknown}; allowed={allowed}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectives", default="all")
    parser.add_argument("--inits", default="all")
    parser.add_argument("--params", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--distill-only", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    objectives = parse_list(args.objectives, OBJECTIVES)
    inits = parse_list(args.inits, INITS)
    params = parse_list(args.params, PARAMS)
    combos = [Combo(obj, init, param) for obj in objectives for init in inits for param in params]
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    if "tm" in objectives and not args.eval_only:
        ensure_expert_trajectory(dry_run=args.dry_run)

    for combo in combos:
        checkpoint = distill_dir(combo) / "full_soft_tokens_best.pth"
        if not args.eval_only and not checkpoint.exists() and parse_eval(combo) is None:
            run_logged(
                distill_command(combo),
                LOG_DIR / f"{combo.run_name}.distill.log",
                dry_run=args.dry_run,
            )
        elif checkpoint.exists() or parse_eval(combo) is not None:
            print(f"skip distill: {combo.run_name}", flush=True)
        else:
            print(f"missing checkpoint for eval-only: {combo.run_name}", flush=True)

        if not args.distill_only and parse_eval(combo) is None:
            if args.dry_run:
                run_logged(
                    eval_command(combo),
                    LOG_DIR / f"{combo.eval_name}.eval.log",
                    dry_run=True,
                )
                continue
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint for eval: {checkpoint}")
            run_logged(
                eval_command(combo),
                LOG_DIR / f"{combo.eval_name}.eval.log",
                dry_run=args.dry_run,
            )
        elif parse_eval(combo) is not None:
            print(f"skip eval: {combo.eval_name}", flush=True)

        if not args.keep_weights and not args.distill_only:
            cleanup_weights(combo)
        updated = collect_results(combos)
        write_results(merge_results(load_existing_results(), updated))

    updated = collect_results(combos)
    write_results(merge_results(load_existing_results(), updated))
    print(f"Wrote {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
