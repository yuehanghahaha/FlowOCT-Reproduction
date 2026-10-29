"""FlowOCT public-code reproduction, pinned to upstream 2309126."""

import os

# Required by deterministic CUDA matmul; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.0.0"
