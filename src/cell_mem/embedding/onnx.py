"""ONNX-based embedding backend — zero PyTorch, zero CUDA.

Uses onnxruntime (~15 MB) + tokenizers (~5 MB) instead of
sentence-transformers + torch (~4.3 GB). Same model weights,
same 384d output quality, 100x smaller disk footprint.

Usage:
    model = ONNXEmbeddingModel("path/to/model.onnx", "path/to/tokenizer/")
    vec = model.embed("hello world")  # -> (384,) float32
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model cache paths
_DEFAULT_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "cell_mem", "onnx"
)
_MODEL_NAME = "all-MiniLM-L6-v2"

# HF mirror for download
_HF_MIRROR = "https://hf-mirror.com"


class ONNXEmbeddingModel:
    """ONNX Runtime embedding model — drop-in replacement for sentence-transformers.

    Loads a pre-exported ONNX model (from PyTorch → ONNX conversion)
    and tokenizer files. No PyTorch or CUDA needed at runtime.
    """

    DIM = 384

    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
    ):
        """
        Args:
            model_path: Path to .onnx file. Auto-detects in cache if None.
            tokenizer_path: Path to tokenizer directory. Auto-detects if None.
        """
        self._model_path = model_path or self._find_onnx_model()
        self._tokenizer_path = tokenizer_path or self._find_tokenizer()
        self._session: Optional[object] = None  # onnxruntime.InferenceSession
        self._tokenizer: Optional[object] = None  # tokenizers.Tokenizer

    # ------------------------------------------------------------------
    # Public API (same interface as EmbeddingModel)
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def ensure_loaded(self) -> None:
        """Load ONNX model and tokenizer if not already loaded."""
        if self._session is not None:
            return

        if not self._model_path or not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {self._model_path}. "
                f"Run 'python -m cell_mem.export_onnx' to export the model, "
                f"or install sentence-transformers for PyTorch fallback."
            )

        import onnxruntime as ort

        # Suppress onnxruntime verbose logging
        ort.set_default_logger_severity(3)  # ERROR only

        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._load_tokenizer()
        logger.info("ONNX model loaded (dim=%d, providers=%s)",
                     self.DIM, self._session.get_providers())

    def embed(self, text: str) -> np.ndarray:
        """Encode single text → (384,) float32 array."""
        self.ensure_loaded()
        inputs = self._tokenize(text)
        outputs = self._session.run(None, inputs)
        vec = outputs[0][0].astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch → (N, 384) float32 array."""
        self.ensure_loaded()
        vectors = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            inputs = self._tokenize(text)
            outputs = self._session.run(None, inputs)
            vec = outputs[0][0].astype(np.float32)
            vectors[i] = vec / (np.linalg.norm(vec) + 1e-10)
        return vectors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> dict:
        """Tokenize text → ONNX-compatible input dict."""
        if self._tokenizer is None:
            self._load_tokenizer()

        encoding = self._tokenizer.encode(text)
        # Truncate to max 128 tokens (same as sentence-transformers default)
        max_len = min(len(encoding.ids), 128)
        input_ids = encoding.ids[:max_len]
        attention_mask = [1] * max_len
        token_type_ids = [0] * max_len

        return {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _load_tokenizer(self) -> None:
        """Load tokenizer from file."""
        from tokenizers import Tokenizer

        tok_path = os.path.join(self._tokenizer_path, "tokenizer.json")
        if os.path.exists(tok_path):
            self._tokenizer = Tokenizer.from_file(tok_path)
            logger.debug("Tokenizer loaded from %s", tok_path)
        else:
            # Fallback: download from HuggingFace
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                f"sentence-transformers/{_MODEL_NAME}"
            )
            logger.info("Tokenizer loaded from HuggingFace (transformers fallback)")

    def _find_onnx_model(self) -> Optional[str]:
        """Auto-detect ONNX model file in cache."""
        candidate = os.path.join(_DEFAULT_CACHE, "model.onnx")
        if os.path.exists(candidate):
            return candidate
        return None

    def _find_tokenizer(self) -> str:
        """Auto-detect tokenizer directory in cache."""
        candidate = os.path.join(_DEFAULT_CACHE)
        tok_file = os.path.join(candidate, "tokenizer.json")
        if os.path.exists(tok_file):
            return candidate
        return candidate  # Will trigger HF download


def export_onnx_model(
    model_name: str = _MODEL_NAME,
    output_dir: str | None = None,
) -> str:
    """Export a sentence-transformers model to ONNX format (one-time operation).

    Requires PyTorch + sentence-transformers (used once for export, then
    can be removed). Produces:
      - model.onnx          (~90 MB, the ONNX graph)
      - tokenizer.json      (~700 KB, tokenizer config)
      - special_tokens_map.json
      - config.json

    Args:
        model_name: HuggingFace model ID.
        output_dir: Output directory. Defaults to ~/.cache/cell_mem/onnx/

    Returns:
        Path to the exported model.onnx file.
    """
    import json
    import shutil

    output_dir = output_dir or _DEFAULT_CACHE
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "model.onnx")
    if os.path.exists(onnx_path):
        logger.info("ONNX model already exists at %s", onnx_path)
        return onnx_path

    logger.info("Exporting %s → ONNX (one-time, requires PyTorch)...", model_name)

    # Requires PyTorch — imported here so ONNX runtime doesn't need it
    import torch
    from sentence_transformers import SentenceTransformer

    # Load the PyTorch model
    model = SentenceTransformer(model_name, device="cpu")
    # Get the underlying transformer for export
    transformer = model._first_module()
    tokenizer = transformer.tokenizer

    # Dummy input for tracing
    dummy_text = "test embedding export"
    inputs = tokenizer(
        dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    # Export to ONNX
    torch.onnx.export(
        transformer.auto_model,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])),
        ),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )

    # Save tokenizer files
    tokenizer.save_pretrained(output_dir)
    # Also save as tokenizer.json for onnx runtime (tokenizers library)
    try:
        tok_json = tokenizer._tokenizer.to_str()
        with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write(tok_json)
    except Exception:
        pass  # Already saved by save_pretrained

    # Save config
    config = {"model_name": model_name, "dim": 384, "max_seq_len": 128}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info("ONNX model exported: %s (%.1f MB)", onnx_path, size_mb)
    return onnx_path
