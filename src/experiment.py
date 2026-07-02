import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Callable
from sklearn.metrics import f1_score, silhouette_score, confusion_matrix
from sklearn.preprocessing import normalize

from encoder import Encoder
from classifier import KNNClassifier
from data_prep import TEXT_COL, LABEL_COL
from data_split import split_base_novel, split_train_test, split_train_val_test

MAX_SHOTS = 10
N_SHOTS = 5
K_NEIGHBORS = 5
BASE_SIZE = 0.6


@dataclass
class Session:
    session: int
    f_macro: float
    f_base: float
    f_novel: float | None               # None for session 0
    base_to_novel_error: float | None   # None for session 0
    novel_to_base_error: float | None   # None for session 0
    silhouette: float
    dbi: float
    confusion_matrix: list[list[int]]
    confusion_labels: list


@dataclass
class Result:
    sessions: list[Session]
    f_bar: float
    perf_drop: float


def _cosine_dbi(X_norm: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    centroids = normalize(np.vstack([
        X_norm[labels == c].mean(axis=0) for c in classes
    ]))
    scatter = np.array([
        float(np.mean(1.0 - X_norm[labels == c] @ centroids[i]))
        for i, c in enumerate(classes)
    ])
    centroid_dist = 1.0 - centroids @ centroids.T
    np.fill_diagonal(centroid_dist, np.inf)  # avoid self-division
    db_ratios = (scatter[:, None] + scatter[None, :]) / centroid_dist
    np.fill_diagonal(db_ratios, -np.inf)    # exclude self from max
    return float(np.mean(db_ratios.max(axis=1)))


def _run_sessions(
    encoder: Encoder,
    df: pd.DataFrame,
    seed: int,
    n_shots: int,
    k_neighbors: int,
    base_size: float,
) -> Result:
    base_df, novel_df = split_base_novel(df, base_size=base_size, random_state=seed)
    base_train_df, _, base_test_df = split_train_val_test(base_df, random_state=seed)

    novel_classes = sorted(novel_df[LABEL_COL].unique())
    rng = np.random.default_rng(seed)
    novel_classes = rng.permutation(novel_classes).tolist()

    # --- Base session ---
    base_classes = sorted(base_train_df[LABEL_COL].unique())

    ref_emb = encoder.embed(base_train_df[TEXT_COL].tolist())
    ref_labels = base_train_df[LABEL_COL].to_numpy()

    base_test_emb = encoder.embed(base_test_df[TEXT_COL].tolist())
    base_test_labels = base_test_df[LABEL_COL].to_numpy()

    clf = KNNClassifier(k=k_neighbors)
    clf.fit(ref_emb, ref_labels)
    base_preds_0 = clf.predict(base_test_emb)
    f0 = f1_score(
        base_test_labels,
        base_preds_0,
        average="macro",
        zero_division=0,
    )
    sil0 = float(silhouette_score(base_test_emb, base_test_labels, metric="cosine"))
    dbi0 = _cosine_dbi(normalize(base_test_emb), base_test_labels)
    cm0 = confusion_matrix(base_test_labels, base_preds_0, labels=base_classes).tolist()

    sessions: list[Session] = [
        Session(session=0, f_macro=f0, f_base=f0, f_novel=None, base_to_novel_error=None, novel_to_base_error=None,
        silhouette=sil0, dbi=dbi0, confusion_matrix=cm0, confusion_labels=[str(l) for l in base_classes])
    ]

    # --- Incremental sessions ---
    novel_test_embs: list[np.ndarray] = []
    novel_test_labels: list[np.ndarray] = []
    seen_novel: list = []

    for t, cls in enumerate(novel_classes, start=1):
        cls_df = novel_df[novel_df[LABEL_COL] == cls]
        shots_df, cls_test_df = split_train_test(cls_df, MAX_SHOTS, n_shots=n_shots, random_state=seed)
        seen_novel.append(cls)

        ref_emb = np.vstack([ref_emb, encoder.embed(shots_df[TEXT_COL].tolist())])
        ref_labels = np.concatenate([ref_labels, shots_df[LABEL_COL].to_numpy()])

        novel_test_embs.append(encoder.embed(cls_test_df[TEXT_COL].tolist()))
        novel_test_labels.append(cls_test_df[LABEL_COL].to_numpy())

        cumul_test_emb = np.vstack([base_test_emb] + novel_test_embs)
        cumul_test_labels = np.concatenate([base_test_labels] + novel_test_labels)

        clf = KNNClassifier(k=k_neighbors)
        clf.fit(ref_emb, ref_labels)
        preds = clf.predict(cumul_test_emb)

        n_base = len(base_test_labels)
        base_preds = preds[:n_base]
        novel_preds = preds[n_base:]
        novel_true = np.concatenate(novel_test_labels)

        silhouette = float(silhouette_score(cumul_test_emb, cumul_test_labels, metric="cosine"))
        dbi = _cosine_dbi(normalize(cumul_test_emb), cumul_test_labels)
        session_labels = base_classes + seen_novel
        cm = confusion_matrix(cumul_test_labels, preds, labels=session_labels).tolist()

        sessions.append(Session(
            session=t,
            f_macro=f1_score(cumul_test_labels, preds, average="macro", zero_division=0),
            f_base=f1_score(base_test_labels, base_preds, average="macro", labels=base_classes, zero_division=0),
            f_novel=f1_score(novel_true, novel_preds, average="macro", labels=seen_novel, zero_division=0),
            base_to_novel_error=float(np.mean(np.isin(base_preds, seen_novel))),
            novel_to_base_error=float(np.mean(np.isin(novel_preds, base_classes))),
            silhouette=silhouette,
            dbi=dbi,
            confusion_matrix=cm,
            confusion_labels=[str(l) for l in session_labels],
        ))

    # --- Aggregate and geometry metrics ---
    f_bar = float(np.mean([s.f_macro for s in sessions]))
    perf_drop = sessions[0].f_macro - sessions[-1].f_macro

    return Result(
        sessions=sessions,
        f_bar=f_bar,
        perf_drop=perf_drop,
    )


def experiment(
    df: pd.DataFrame,
    get_encoder: Callable[[int], Encoder],
    seeds: list[int],
    sweep_vals: list[int],
    base_size: float = BASE_SIZE,
) -> dict[str, dict[int, list[Result]]]:
    results: dict[str, dict[int, list[Result]]] = {
        "k_neighbors": {val: [] for val in sweep_vals},
        "n_shots": {val: [] for val in sweep_vals},
    }
    for seed in seeds:
        encoder = get_encoder(seed)
        for val in sweep_vals:
            results["k_neighbors"][val].append(
                _run_sessions(encoder, df, seed, N_SHOTS, val, base_size)
            )
            results["n_shots"][val].append(
                _run_sessions(encoder, df, seed, val, K_NEIGHBORS, base_size)
            )
    return results
