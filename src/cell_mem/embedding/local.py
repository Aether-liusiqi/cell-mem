"""Local embedding model and Johnson-Lindenstrauss random projection matrix.

EmbeddingModel wraps sentence-transformers (all-MiniLM-L6-v2, 384d).
ProjectionMatrix performs 384d → 2048d sparse projection for pattern separation
(imitating the dentate gyrus's ability to orthogonalize similar inputs).
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace mirror for environments where huggingface.co is blocked
# ---------------------------------------------------------------------------
_HF_MIRROR = "https://hf-mirror.com"


def _ensure_hf_mirror() -> None:
    """Set HF endpoint to mirror if not already configured."""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = _HF_MIRROR
        logger.info("HF_ENDPOINT set to %s", _HF_MIRROR)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ---------------------------------------------------------------------------
# EmbeddingModel
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Thin wrapper around sentence-transformers for text → vector encoding.

    On first use, downloads ~90MB model to the HuggingFace cache. Subsequent
    loads are instant. The model is loaded lazily — call ensure_loaded() to
    force eager loading, or let embed() trigger it automatically.

    Usage:
        model = EmbeddingModel()
        model.ensure_loaded()      # optional, pre-loads
        vec = model.embed("hello") # -> (384,) float32 array
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DIM = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        preload: bool = False,
    ):
        _ensure_hf_mirror()
        self._model_name = model_name
        self._device = device
        self._model: Optional[object] = None  # SentenceTransformer
        if preload:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """Load the model if not already loaded. Blocks ~2-5s on first call."""
        if self._model is not None:
            return
        t0 = time.time()
        # Suppress verbose sentence-transformers/huggingface output during loading.
        # These emit 100+ lines of download progress, tokenizer config, and model
        # architecture details that flood stderr and can deadlock subprocess pipes.
        import io
        import sys
        import warnings
        from contextlib import redirect_stderr, redirect_stdout

        from sentence_transformers import SentenceTransformer

        # Silence transformers/model loading chatter at the logger level
        for noisy_logger in (
            "sentence_transformers", "transformers", "tokenizers",
            "huggingface_hub", "filelock",
        ):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        # Also suppress tqdm progress bars and stdout noise during loading.
        # SentenceTransformer prints BERT LOAD REPORT tables and download bars
        # that can't be caught at the logger level.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            devnull = io.StringIO()
            try:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    logger.info("Loading embedding model: %s ...", self._model_name)
                    self._model = SentenceTransformer(self._model_name, device=self._device)
            except Exception:
                # If redirect fails (e.g. IPython), load without suppression
                self._model = SentenceTransformer(self._model_name, device=self._device)

        # Restore log levels
        for noisy_logger in (
            "sentence_transformers", "transformers", "tokenizers",
            "huggingface_hub", "filelock",
        ):
            logging.getLogger(noisy_logger).setLevel(logging.NOTSET)

        elapsed = time.time() - t0
        # Use get_embedding_dimension (new API) with fallback to old method name
        try:
            dim = self._model.get_embedding_dimension()
        except AttributeError:
            dim = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded in %.1fs (dim=%d)", elapsed, dim)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def embed(self, text: str) -> np.ndarray:
        """Encode a single text → (384,) float32 array."""
        self.ensure_loaded()
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch → (N, 384) float32 array."""
        self.ensure_loaded()
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Alias for embed() — used by recall for query vectors."""
        return self.embed(text)


# ---------------------------------------------------------------------------
# ProjectionMatrix
# ---------------------------------------------------------------------------


class ProjectionMatrix:
    """384d → 2048d random projection for pattern separation.

    Uses the Johnson-Lindenstrauss lemma: a random matrix preserves pairwise
    distances while expanding dimensionality. We then sparse-code the result
    (keep top 5% by absolute value) to simulate dentate gyrus sparsity (~5%).

    The projection matrix (~3MB) is generated once (seed=42 for reproducibility)
    and persisted in the SQLite meta table so all instances share the same mapping.

    Usage:
        proj = ProjectionMatrix(sqlite_store)
        sparse_2048d = proj.project(embedding_384d)
    """

    INPUT_DIM = 384
    OUTPUT_DIM = 2048
    SPARSITY = 0.05  # retain top 5% of dimensions

    def __init__(self, sqlite_store: "SqliteStore"):  # noqa: F821
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = sqlite_store
        self._matrix: Optional[np.ndarray] = None
        self._load_or_generate()

    def _load_or_generate(self) -> None:
        """Load projection matrix from SQLite meta, or generate + persist it."""
        blob = self._store.get_meta("projection_matrix")
        if blob is not None:
            self._matrix = self._deserialize(blob)
            logger.info(
                "Loaded projection matrix from DB: (%d, %d)",
                self._matrix.shape[0],
                self._matrix.shape[1],
            )
        else:
            self._matrix = self._generate()
            self._store.set_meta(
                "projection_matrix", self._serialize(self._matrix)
            )
            logger.info(
                "Generated and stored projection matrix: (%d, %d)",
                self._matrix.shape[0],
                self._matrix.shape[1],
            )

    def _generate(self) -> np.ndarray:
        """Generate Johnson-Lindenstrauss random projection matrix.

        Variance is 1/OUTPUT_DIM so the dot product has unit variance.
        Seed=42 ensures reproducibility across instances.
        """
        rng = np.random.RandomState(42)
        matrix = rng.randn(self.OUTPUT_DIM, self.INPUT_DIM).astype(np.float32)
        matrix *= np.sqrt(1.0 / self.OUTPUT_DIM)
        return matrix

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project a 384d embedding to a sparse 2048d vector.

        1. Matrix multiply: (2048, 384) @ (384,) → (2048,)
        2. Take absolute values
        3. Keep top-5% values, zero the rest
        4. Return sparse vector (same scale, ~102 non-zero dims)
        """
        if self._matrix is None:
            raise RuntimeError("Projection matrix not initialized")

        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.shape[0] != self.INPUT_DIM:
            raise ValueError(
                f"Expected {self.INPUT_DIM}d input, got {vec.shape[0]}d"
            )

        # Step 1: matrix multiplication
        projected = self._matrix @ vec  # (2048,)

        # Step 2-3: sparsify — keep top k by absolute value
        k = max(1, int(self.OUTPUT_DIM * self.SPARSITY))  # ~102
        abs_proj = np.abs(projected)
        top_k_idx = np.argpartition(abs_proj, -k)[-k:]  # indices of k largest
        mask = np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        mask[top_k_idx] = 1.0

        sparse = projected * mask

        # Normalize so cosine similarity remains meaningful
        norm = np.linalg.norm(sparse)
        if norm > 1e-8:
            sparse /= norm

        return sparse

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(matrix: np.ndarray) -> bytes:
        import sqlite_vec

        return sqlite_vec.serialize_float32(matrix.ravel().tolist())

    @staticmethod
    def _deserialize(blob: bytes) -> np.ndarray:
        # sqlite_vec.serialize_float32 uses flat float32 binary format
        flat = np.frombuffer(blob, dtype=np.float32)
        return flat.reshape(ProjectionMatrix.OUTPUT_DIM, ProjectionMatrix.INPUT_DIM)
