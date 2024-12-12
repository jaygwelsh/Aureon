import numpy as np

def custom_objective(x: np.ndarray) -> float:
    return np.sum(x**2)  # Simple sphere function

def custom_gradient(x: np.ndarray) -> np.ndarray:
    return 2*x
