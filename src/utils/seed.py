"""Global seeding utility for reproducibility."""
from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, numpy, and torch (CPU + CUDA).

    Torch is imported lazily so this module stays cheap to import in
    CPU-only contexts (e.g. the data diagnostics).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
