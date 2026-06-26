import argparse
import json
import pandas as pd

from pathlib import Path
from classifier import KNNClassifier
from encoder import Encoder
from fine_tune import optimize_hyperparameters, fine_tune
from data_prep import load_wos_dataset, load_hr_dataset, TEXT_COL, LABEL_COL
from data_split import split_base_novel, split_train_val_test
from sklearn.metrics import f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
SEED = 42
HR = "HumanResources"
WOS = "WebOfScience"


def run_hpo(dataset_name: str, df: pd.DataFrame, output_dir: Path) -> None:  
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df, _ = split_base_novel(df, random_state=SEED)
    base_train_df, base_val_df, _ = split_train_val_test(
        base_df, random_state=SEED
    )

    base_classes = sorted(base_df["label"].unique().tolist())
    print(f"[{dataset_name}] Base classes ({len(base_classes)}): {', '.join(str(c) for c in base_classes)}")

    encoder = Encoder()

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
        json.dump({
            "seed": SEED,
            "base_classes": base_classes,
            "best_params": best_params,
            "best_value": study.best_value,
        }, f, indent=2)
    print(f"[{dataset_name}] Best hyperparameters saved to {output_dir / 'hpo_best_params.json'}")


def run_fine_tune(dataset_name: str, df: pd.DataFrame, output_dir: Path) -> None:
    params_path = output_dir / dataset_name / "hpo_best_params.json"
    hpo_results = _load_hpo_results(params_path)

    print(f"[{dataset_name}] Loaded best params: {hpo_results['best_params']}")

    base_df, _ = split_base_novel(df, random_state=SEED)
    base_train_df, base_val_df, _ = split_train_val_test(base_df, random_state=SEED)

    encoder = Encoder()
    fine_tune(encoder, base_train_df, seed=SEED, **hpo_results['best_params'])
    print(f"[{dataset_name}] Fine-tuning complete.")

    model_path = output_dir / dataset_name / "fine_tuned_encoder"
    encoder.save_model(str(model_path))
    print(f"[{dataset_name}] Encoder saved to {model_path}")

    if hpo_results["seed"] == SEED:
        train_emb = encoder.embed(base_train_df[TEXT_COL].tolist())
        val_emb = encoder.embed(base_val_df[TEXT_COL].tolist())

        clf = KNNClassifier()
        clf.fit(train_emb, base_train_df[LABEL_COL].to_numpy())
        preds = clf.predict(val_emb)
        score = f1_score(base_val_df[LABEL_COL].to_numpy(), preds, average="macro", zero_division=0)
        print(f"[{dataset_name}] Macro-F1: {score:.4f}")

        if hpo_results["best_value"] != score:
            print(f"Macro-F1 does not match expected value of hpo run: {hpo_results["best_value"]:.4f}")


def _load_hpo_results(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No HPO results found at '{path}'. Run with --mode hpo first."
        )
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wos-path", type=Path)
    parser.add_argument("--hr-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "output")
    parser.add_argument("--mode", choices=["hpo", "fine-tune"], default="fine-tune")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wos_path:
        wos_df = load_wos_dataset(args.wos_path)
        if args.mode == "hpo":
            run_hpo(WOS, wos_df, args.output_dir)
        else:
            run_fine_tune(WOS, wos_df, args.output_dir)

    if args.hr_path:
        hr_df = load_hr_dataset(args.hr_path)
        if args.mode == "hpo":
            run_hpo(HR, hr_df, args.output_dir)
        else:
            run_fine_tune(HR, hr_df, args.output_dir)


if __name__ == "__main__":
    main()
