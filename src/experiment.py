import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import f1_score, silhouette_score
from sklearn.preprocessing import normalize

from encoder import Encoder
from classifier import KNNClassifier
from data_prep import TEXT_COL, LABEL_COL
from data_split import split_base_novel, split_train_test, split_train_val_test


@dataclass
class Session:
    session: int
    f_macro: float
    f_base: float
    f_novel: float | None               # None for session 0
    base_to_novel_error: float | None   # None for session 0


@dataclass
class Result:
    sessions: list[Session]
    f_bar: float
    perf_drop: float
    silhouette: float
    dbi: float


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


def experiment(
    encoder: Encoder,
    df: pd.DataFrame,
    n_shots: int = 5,
    k_neighbors: int = 5,
    n_novel_classes: int | None = None,
    base_size: float = 0.6,
    seed: int = 42,
) -> Result:
    base_df, novel_df = split_base_novel(df, base_size=base_size, random_state=seed)
    base_train_df, _, base_test_df = split_train_val_test(base_df, random_state=seed)

    novel_classes = sorted(novel_df[LABEL_COL].unique())
    rng = np.random.default_rng(seed)
    novel_classes = rng.permutation(novel_classes).tolist()
    if n_novel_classes is not None:
        novel_classes = novel_classes[:n_novel_classes]

    # --- Session 0: base only ---
    base_classes = sorted(base_train_df[LABEL_COL].unique())

    ref_emb = encoder.embed(base_train_df[TEXT_COL].tolist())
    ref_labels = base_train_df[LABEL_COL].to_numpy()

    base_test_emb = encoder.embed(base_test_df[TEXT_COL].tolist())
    base_test_labels = base_test_df[LABEL_COL].to_numpy()

    clf = KNNClassifier(k=k_neighbors)
    clf.fit(ref_emb, ref_labels)
    f0 = f1_score(
        base_test_labels,
        clf.predict(base_test_emb),
        average="macro",
        zero_division=0,
    )

    sessions: list[Session] = [
        Session(session=0, f_macro=f0, f_base=f0, f_novel=None, base_to_novel_error=None)
    ]

    # --- Incremental sessions ---
    novel_test_embs: list[np.ndarray] = []
    novel_test_labels: list[np.ndarray] = []
    seen_novel: list = []

    for t, cls in enumerate(novel_classes, start=1):
        cls_df = novel_df[novel_df[LABEL_COL] == cls]
        shots_df, cls_test_df = split_train_test(cls_df, n_shots=n_shots, random_state=seed)
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

        sessions.append(Session(
            session=t,
            f_macro=f1_score(cumul_test_labels, preds, average="macro", zero_division=0),
            f_base=f1_score(base_test_labels, base_preds, average="macro", labels=base_classes, zero_division=0),
            f_novel=f1_score(novel_true, novel_preds, average="macro", labels=seen_novel, zero_division=0),
            base_to_novel_error=float(np.mean(np.isin(base_preds, seen_novel))),
        ))

    # --- Aggregate & geometry metrics ---
    f_bar = float(np.mean([s.f_macro for s in sessions]))
    perf_drop = sessions[0].f_macro - sessions[-1].f_macro

    silhouette = float(silhouette_score(ref_emb, ref_labels, metric="cosine"))
    dbi = _cosine_dbi(normalize(ref_emb), ref_labels)

    return Result(
        sessions=sessions,
        f_bar=f_bar,
        perf_drop=perf_drop,
        silhouette=silhouette,
        dbi=dbi,
    )
