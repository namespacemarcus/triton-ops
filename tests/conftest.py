import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return "cuda"
