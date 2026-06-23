import argparse
import json
import pandas as pd

from pathlib import Path
from encoder import Encoder
from fine_tune import optimize_hyperparameters
from data_prep import load_wos_dataset, load_hr_dataset
from data_split import split_base_novel, split_train_val_test

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
MAX_SEQ_LENGTH = 512


def run_hpo(dataset_name: str, df: pd.DataFrame, output_dir: Path) -> None:
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df, _ = split_base_novel(df, random_state=SEED)
    base_train_df, base_val_df, _ = split_train_val_test(
        base_df, random_state=SEED
    )

    encoder = Encoder(MODEL_NAME, MAX_SEQ_LENGTH)

    study = optimize_hyperparameters(
        encoder,
        base_train_df,
        base_val_df,
        seed=SEED,
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "hpo_trials.csv", index=False)
    print(f"[{dataset_name}] All trials saved to {output_dir / 'hpo_trials.csv'}")

    best_params = study.best_params
    with open(output_dir / "hpo_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[{dataset_name}] Best params (macro-F1 {study.best_value:.4f}): {best_params}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wos-path", type=Path)
    parser.add_argument("--hr-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "output")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wos_path:
        wos_df = load_wos_dataset(args.wos_path)
        run_hpo("wos", wos_df, args.output_dir)

    if args.hr_path:
        hr_df = load_hr_dataset(args.hr_path)
        run_hpo("hr", hr_df, args.output_dir)


if __name__ == "__main__":
    main()
