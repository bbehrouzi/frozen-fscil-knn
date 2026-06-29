import pandas as pd

from sklearn.model_selection import train_test_split
from data_prep import LABEL_COL


def split_base_novel(
    df: pd.DataFrame,
    base_size: float = 0.6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_classes = df[LABEL_COL].unique()
    n_base_classes = round(len(all_classes) * base_size)

    rng = pd.Series(all_classes).sample(frac=1, random_state=random_state)
    base_classes = set(rng.iloc[:n_base_classes])
    novel_classes = set(rng.iloc[n_base_classes:])

    base_df = df[df[LABEL_COL].isin(base_classes)].reset_index(drop=True)
    novel_df = df[df[LABEL_COL].isin(novel_classes)].reset_index(drop=True)

    return base_df, novel_df


def split_train_val_test(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[LABEL_COL],
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[LABEL_COL],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_train_test(
        df: pd.DataFrame,
        max_shots: int, 
        n_shots: int = 5,
        random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert df[LABEL_COL].nunique() == 1, "split_train_test expects a single-class frame"
    pool = df.sample(n=max_shots, random_state=random_state)
    test_df = df.drop(index=pool.index).reset_index(drop=True)
    train_df = pool.head(n_shots).reset_index(drop=True)
    return train_df, test_df
