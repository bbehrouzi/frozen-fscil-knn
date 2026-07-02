import argparse
import json
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from encoder import Encoder
from fine_tune import hp_optimization, fine_tune
from experiment import experiment, Result, K_NEIGHBORS
from data_prep import load_wos_dataset, load_hr_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    mode: str
    data_path: Path
    seeds: list[int]
    out_dir: Path | None = None
    sweep_vals: list[int] | None = None


def run_hp_optimization(df: pd.DataFrame, cfg: Config):

    study, base_classes = hp_optimization(
        df,
        seed=cfg.seeds[0],
        k_neighbors=K_NEIGHBORS,
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(cfg.out_dir / "hpo_trials.csv", index=False)
    print(f"All trials saved to {cfg.out_dir / 'hpo_trials.csv'}")

    with open(cfg.out_dir / "hpo_summary.json", "w") as f:
        json.dump({
            "seed": cfg.seeds[0],
            "base_classes": base_classes,
            "best_params": study.best_params,
            "best_value": study.best_value,
        }, f, indent=2)
    print(f"[Best hyperparameters saved to {cfg.out_dir / 'hpo_summary.json'}")


def run_fine_tune(df: pd.DataFrame, cfg: Config):
    hpo_summary = load_hpo_summary(cfg.out_dir / "hpo_summary.json")
    print(f"[Loaded best params: {hpo_summary['best_params']}")

    for seed in cfg.seeds:
        encoder = Encoder()
        fine_tune(encoder, df, seed=seed, **hpo_summary["best_params"])
        model_path = cfg.out_dir / f"seed_{seed}" / "fine_tuned_encoder"
        model_path.mkdir(parents=True, exist_ok=True)
        encoder.save_model(str(model_path))
        print(f"[seed={seed}: encoder saved to {model_path}")


def _save_experiment_results(
    results: dict[str, dict[int, list[Result]]],
    seeds: list[int],
    out_dir: Path,
    encoder_tag: str,
) -> None:
    for sweep_var, val_results in results.items():
        for sweep_val, seed_results in val_results.items():
            for seed, result in zip(seeds, seed_results):
                seed_dir = out_dir / f"seed_{seed}" / encoder_tag / f"{sweep_var}_{sweep_val}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                confusions = [
                    {"session": s.session, "labels": s.confusion_labels, "matrix": s.confusion_matrix}
                    for s in result.sessions
                ]
                with open(seed_dir / "eval_confusions.json", "w") as f:
                    json.dump(confusions, f, indent=2)

                sessions_data = [
                    {k: v for k, v in s.__dict__.items() if k not in ("confusion", "confusion_labels")}
                    for s in result.sessions
                ]
                pd.DataFrame(sessions_data).to_csv(seed_dir / "eval_sessions.csv", index=False)

                with open(seed_dir / "eval_summary.json", "w") as f:
                    json.dump({"f_bar": result.f_bar, "perf_drop": result.perf_drop}, f, indent=2)

                print(f"seed={seed} {sweep_var}={sweep_val}: saved to {seed_dir}")


def run_experiment(df: pd.DataFrame, cfg: Config) -> None:
    print("Running experiment with fine-tuned encoder...")
    ft_results = experiment(
        df, lambda seed: Encoder(str(cfg.out_dir / f"seed_{seed}" / "fine_tuned_encoder")),
        cfg.seeds, cfg.sweep_vals,
    )
    _save_experiment_results(ft_results, cfg.seeds, cfg.out_dir, "fine_tuned")

    print("Running experiment with pretrained encoder...")
    pt_encoder = Encoder()
    pt_results = experiment(df, lambda _: pt_encoder, cfg.seeds, cfg.sweep_vals)
    _save_experiment_results(pt_results, cfg.seeds, cfg.out_dir, "pretrained")


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
    return Config(
        mode=data["mode"],
        data_path=Path(data["data_path"]),
        seeds=data["seeds"],
        out_dir=Path(data["out_dir"]) if "out_dir" in data else None,
        sweep_vals=data.get("sweep_vals"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config JSON file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data_path.suffix == ".csv":
        df = load_hr_dataset(cfg.data_path)
    else:
        df = load_wos_dataset(cfg.data_path)

    if cfg.mode == "hp-optimization":
        run_hp_optimization(df, cfg)
    elif cfg.mode == "fine-tune":
        run_fine_tune(df, cfg)
    elif cfg.mode == "experiment":
        run_experiment(df, cfg)


if __name__ == "__main__":
    main()
