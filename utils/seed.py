import random
import numpy as np
import torch

from utils.config import Config


def set_seed(config: Config, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.system.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Override backend if user requested determinism
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True