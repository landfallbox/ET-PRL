from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def configure_reproducibility(
    seed: int,
    *,
    deterministic_cudnn: bool = False,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic_cudnn)
        torch.backends.cudnn.benchmark = not bool(deterministic_cudnn)

    return {
        "seed": int(seed),
        "deterministic_cudnn": bool(deterministic_cudnn),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False))
        if hasattr(torch.backends, "cudnn")
        else False,
        "cuda_available": bool(torch.cuda.is_available()),
    }
