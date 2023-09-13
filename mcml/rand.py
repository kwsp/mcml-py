import numpy as np
from numba import njit


@njit
def get_random() -> float:
    """
    Make sure >0
    """
    return np.random.random() + 1e-9
    # return 0.5
