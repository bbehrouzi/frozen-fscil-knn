import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid


class KNNClassifier:
    def __init__(self, k: int = 5) -> None:
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


class NCMClassifier:
    def __init__(self) -> None:
        self._clf = NearestCentroid(metric="cosine")

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self._clf.fit(embeddings, labels)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._clf.predict(embeddings)
