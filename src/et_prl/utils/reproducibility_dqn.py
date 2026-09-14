from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

try:
    from et_prl.utils import configure_reproducibility as _toolkit_configure_reproducibility
except ImportError:
    _toolkit_configure_reproducibility = None


def configure_reproducibility(
    seed: int,
    *,
    deterministic_cudnn: bool = False,
) -> dict[str, Any]:
    if _toolkit_configure_reproducibility is not None:
        return _toolkit_configure_reproducibility(
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
        )

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
