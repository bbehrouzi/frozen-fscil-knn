import numpy as np

from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._clf = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
            weights="distance",
        )

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self._clf.fit(embeddings, labels)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._clf.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(embeddings)
