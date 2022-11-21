import numpy as np
import numba


@numba.jit
def get_random() -> float:
    """
    Make sure >0
    """
    return np.random.random() + 1e-9
