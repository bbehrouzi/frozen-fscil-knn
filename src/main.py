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
    n_novel_classes: int
    base_size: float


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
    all_results = []

    for seed in cfg.seeds:
        encoder_path = output_dir / dataset_name / f"seed_{seed}" / "fine_tuned_encoder"
        encoder = Encoder()
        if encoder_path.exists():
            encoder.load_model(encoder_path)
            print(f"[{dataset_name}] seed={seed}: loaded fine-tuned encoder from {encoder_path}")
        else:
            print(f"[{dataset_name}] seed={seed}: no fine-tuned encoder found, using pre-trained encoder")

        result = experiment(
            encoder,
            df,
            n_shots=cfg.n_shots,
            k_neighbors=cfg.k_neighbors,
            n_novel_classes=cfg.n_novel_classes,
            base_size=cfg.base_size,
            seed=seed,
        )
        all_results.append(result)

        seed_dir = output_dir / dataset_name / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        sessions_df = pd.DataFrame([s.__dict__ for s in result.sessions])
        sessions_df.to_csv(seed_dir / "eval_sessions.csv", index=False)
        with open(seed_dir / "eval_summary.json", "w") as f:
            json.dump({
                "f_bar": result.f_bar,
                "perf_drop": result.perf_drop,
                "silhouette": result.silhouette,
                "dbi": result.dbi,
            }, f, indent=2)
        print(f"[{dataset_name}] seed={seed}: results saved to {seed_dir}")

    # Average scalar metrics across seeds
    eval_dir = output_dir / dataset_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    scalar_keys = ["f_bar", "perf_drop", "silhouette", "dbi"]
    agg: dict = {"seeds": cfg.seeds}
    for k in scalar_keys:
        vals = [getattr(r, k) for r in all_results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    with open(eval_dir / "eval_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[{dataset_name}] Aggregate metrics (mean +/- std over {len(cfg.seeds)} seeds): {agg}")


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
        n_novel_classes=data["n_novel_classes"],
        base_size=data["base_size"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config JSON file")
    cfg = load_config(parser.parse_args().config)

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
