import numpy as np

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 512


class Encoder:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model = SentenceTransformer(MODEL_NAME)
        self.model.max_seq_length = MAX_SEQ_LENGTH
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

    def update_model(self, new_model: SentenceTransformer):
        self._cache.clear()
        self.model = new_model

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        self.model = SentenceTransformer(path)
