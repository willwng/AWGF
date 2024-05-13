import numpy as np


def sample_distribution(n_dimensions: int, low: float, high: float) -> np.ndarray:
    """
    Sample a random position in n_dimensions from a uniform distribution
    """
    return np.random.uniform(low, high, (n_dimensions,))
