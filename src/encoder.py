import numpy as np

from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(self, model_name: str, max_seq_length: int = 512) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            embeddings = self.model.encode(
                missing,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for text, emb in zip(missing, embeddings):
                self._cache[text] = emb

        return np.stack([self._cache[t] for t in texts])

    def update_model(self, new_model: SentenceTransformer) -> None:
        self._cache.clear()
        self.model = new_model
