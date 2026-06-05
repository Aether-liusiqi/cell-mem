"""Export all-MiniLM-L6-v2 to ONNX format (one-time, requires PyTorch).

Usage:
    python -m cell_mem.export_onnx
    python -m cell_mem.export_onnx --output /path/to/onnx/

After export, the ONNX model is cached to ~/.cache/cell_mem/onnx/
and used automatically by EmbeddingModel. PyTorch is no longer needed.
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cell_mem.export_onnx")


def main():
    parser = argparse.ArgumentParser(
        description="Export all-MiniLM-L6-v2 to ONNX for zero-PyTorch inference"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ~/.cache/cell_mem/onnx/)",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Model name on HuggingFace",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Cell-mem ONNX Model Export")
    print("=" * 60)
    print()
    print("This converts the PyTorch model to ONNX format.")
    print("After export, PyTorch (4.3 GB) can be removed — only")
    print("onnxruntime (~15 MB) + tokenizers (~5 MB) are needed.")
    print()
    print("Export takes ~30s and requires ~90 MB disk for the ONNX file.")
    print()

    from cell_mem.embedding.onnx import export_onnx_model

    output = export_onnx_model(model_name=args.model, output_dir=args.output)

    print()
    print(f"✓ ONNX model exported to: {output}")
    print("✓ Cell-mem will auto-use ONNX on next startup.")
    print("✓ To remove PyTorch: pip uninstall torch sentence-transformers")


if __name__ == "__main__":
    main()
