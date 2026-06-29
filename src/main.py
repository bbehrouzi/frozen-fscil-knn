import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from encoder import Encoder
from fine_tune import hp_optimization, fine_tune
from experiment import experiment
from data_prep import load_wos_dataset, load_hr_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
HR = "HumanResources"
WOS = "WebOfScience"


@dataclass
class Config:
    mode: str
    data_path: Path
    output_dir: Path
    seeds: list[int]
    n_shots: int
    k_neighbors: int
    base_size: float
    use_fine_tuned: bool
    sweep: dict | None = None


def run_hp_optimization(dataset_name: str, df: pd.DataFrame, output_dir: Path, cfg: Config):  
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    study, base_classes = hp_optimization(
        df,
        seed=cfg.seeds[0],
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "hpo_trials.csv", index=False)
    print(f"[{dataset_name}] All trials saved to {output_dir / 'hpo_trials.csv'}")

    with open(output_dir / "hpo_summary.json", "w") as f:
        json.dump({
            "seed": cfg.seeds[0],
            "base_classes": base_classes,
            "best_params": study.best_params,
            "best_value": study.best_value,
        }, f, indent=2)
    print(f"[{dataset_name}] Best hyperparameters saved to {output_dir / 'hpo_summary.json'}")


def run_fine_tune(dataset_name: str, df: pd.DataFrame, output_dir: Path, cfg: Config):
    hpo_summary = load_hpo_summary(output_dir / dataset_name / "hpo_summary.json")
    print(f"[{dataset_name}] Loaded best params: {hpo_summary['best_params']}")

    for seed in cfg.seeds:
        encoder = Encoder()
        fine_tune(encoder, df, seed=seed, **hpo_summary['best_params'])
        model_path = output_dir / dataset_name / f"seed_{seed}" / "fine_tuned_encoder"
        model_path.mkdir(parents=True, exist_ok=True)
        encoder.save_model(str(model_path))
        print(f"[{dataset_name}] seed={seed}: encoder saved to {model_path}")


def run_experiment(dataset_name: str, df: pd.DataFrame, output_dir: Path, cfg: Config):
    sweep_var = cfg.sweep["variable"] if cfg.sweep else None
    sweep_values = cfg.sweep["values"] if cfg.sweep else [None]

    results_by_sweep: dict = {v: [] for v in sweep_values}

    for seed in cfg.seeds:
        if cfg.use_fine_tuned:
            encoder_path = output_dir / dataset_name / f"seed_{seed}" / "fine_tuned_encoder"
            encoder = Encoder(str(encoder_path))
            print(f"[{dataset_name}] seed={seed}: loaded fine-tuned encoder from {encoder_path}")
        else:
            encoder = Encoder()
            print(f"[{dataset_name}] seed={seed}: using pre-trained encoder {encoder.model_name}")

        for sweep_val in sweep_values:
            exp_kwargs = {
                "n_shots": cfg.n_shots,
                "k_neighbors": cfg.k_neighbors,
                "base_size": cfg.base_size,
            }
            if sweep_var is not None:
                exp_kwargs[sweep_var] = sweep_val
                seed_dir = output_dir / dataset_name / f"seed_{seed}" / f"{sweep_var}_{sweep_val}"
            else:
                seed_dir = output_dir / dataset_name / f"seed_{seed}"

            result = experiment(encoder, df, seed=seed, **exp_kwargs)
            results_by_sweep[sweep_val].append(result)

            seed_dir.mkdir(parents=True, exist_ok=True)

            confusions = [
                {"session": s.session, "labels": s.confusion_labels, "matrix": s.confusion}
                for s in result.sessions
            ]
            with open(seed_dir / "eval_confusions.json", "w") as f:
                json.dump(confusions, f, indent=2)

            sessions_data = [
                {k: v for k, v in s.__dict__.items() if k not in ("confusion", "confusion_labels")}
                for s in result.sessions
            ]
            sessions_df = pd.DataFrame(sessions_data)
            sessions_df.to_csv(seed_dir / "eval_sessions.csv", index=False)
            with open(seed_dir / "eval_summary.json", "w") as f:
                json.dump({"f_bar": result.f_bar, "perf_drop": result.perf_drop}, f, indent=2)
            print(f"[{dataset_name}] seed={seed} {f'{sweep_var}={sweep_val}' if sweep_var else ''}: saved to {seed_dir}")

    # Average scalar metrics across seeds per sweep value
    for sweep_val, all_results in results_by_sweep.items():
        if sweep_var is not None:
            agg_dir = output_dir / dataset_name / f"{sweep_var}_{sweep_val}"
        else:
            agg_dir = output_dir / dataset_name
        agg_dir.mkdir(parents=True, exist_ok=True)
        agg: dict = {"seeds": cfg.seeds}
        if sweep_var is not None:
            agg[sweep_var] = sweep_val
        for k in ["f_bar", "perf_drop"]:
            vals = [getattr(r, k) for r in all_results]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        with open(agg_dir / "eval_summary.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[{dataset_name}] {f'{sweep_var}={sweep_val} ' if sweep_var else ''}aggregated over {len(cfg.seeds)} seeds: {agg}")


def load_hpo_summary(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No HPO results found at '{path}'. Run with --mode hpo first."
        )
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_config(path: Path) -> Config:
    with open(path) as f:
        data = json.load(f)
    raw = data.get("seeds", data.get("seed", 42))
    seeds = raw if isinstance(raw, list) else [raw]
    return Config(
        mode=data["mode"],
        data_path=Path(data["data_path"]),
        output_dir=Path(data["output_dir"]),
        seeds=seeds,
        n_shots=data["n_shots"],
        k_neighbors=data["k_neighbors"],
        base_size=data["base_size"],
        use_fine_tuned=data["use_fine_tuned"],
        sweep=data.get("sweep", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config JSON file")
    parser.add_argument("--data-path", type=Path, default=None, help="Override data_path from config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_path is not None:
        cfg.data_path = args.data_path

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    name, loader = (HR, load_hr_dataset) if HR in str(cfg.data_path) else (WOS, load_wos_dataset)
    df = loader(cfg.data_path)

    if cfg.mode == "hp-optimization":
        run_hp_optimization(name, df, cfg.output_dir, cfg)
    elif cfg.mode == "fine-tune":
        run_fine_tune(name, df, cfg.output_dir, cfg)
    else:
        run_experiment(name, df, cfg.output_dir, cfg)


if __name__ == "__main__":
    main()
