import numpy as np


def get_random() -> float:
    """
    Make sure >0
    """
    return np.random.random() + 1e-9
