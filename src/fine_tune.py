import json
from pathlib import Path
import pandas as pd
import optuna

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score

from classifier import KNNClassifier
from encoder import Encoder
from data_prep import TEXT_COL, LABEL_COL


def fine_tune(
    encoder: Encoder,
    train_df: pd.DataFrame,
    seed: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
) -> None:

    def model_init():
        m = SetFitModel.from_pretrained(encoder.model_name)
        m.model_body.max_seq_length = encoder.model.max_seq_length
        return m

    train_dataset = Dataset.from_dict(
        {
            TEXT_COL: train_df[TEXT_COL].tolist(),
            LABEL_COL: train_df[LABEL_COL].tolist(),
        }
    )

    args = TrainingArguments(
        batch_size=batch_size,
        body_learning_rate=learning_rate,
        max_steps=max_steps,
        seed=seed,
    )

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        column_mapping={TEXT_COL: TEXT_COL, LABEL_COL: LABEL_COL},
    )

    trainer.train()
    print("Fine-tuning complete. Updating encoder...")
    encoder.update_model(trainer.model.model_body)
    print("Encoder updated.")


def optimize_hyperparameters(
    encoder: Encoder,
    base_train_df: pd.DataFrame,
    base_val_df: pd.DataFrame,
    n_trials: int = 15,
    seed: int = 42,
) -> optuna.Study:
    
    def objective(trial: optuna.Trial) -> float:
        print(f"Starting trial {trial.number + 1}/{n_trials}...")
        trial_encoder = Encoder(encoder.model_name, encoder.model.max_seq_length)

        fine_tune(
            trial_encoder,
            base_train_df,
            seed=seed,
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            max_steps=trial.suggest_int("max_steps", 250, 2500),
        )

        train_emb = trial_encoder.embed(base_train_df[TEXT_COL].tolist())
        val_emb = trial_encoder.embed(base_val_df[TEXT_COL].tolist())

        clf = KNNClassifier()
        clf.fit(train_emb, base_train_df[LABEL_COL].to_numpy())
        preds = clf.predict(val_emb)
        score = f1_score(base_val_df[LABEL_COL].to_numpy(), preds, average="macro", zero_division=0)
        print(f"Trial {trial.number + 1} macro-F1: {score:.4f}")
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study
