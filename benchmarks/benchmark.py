import time
import numpy as np
from subprocess import check_output
from scipy.optimize import minimize

# Example benchmark: compare Aureon vs SciPy minimize on a chosen objective (e.g., Rosenbrock)
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def run_aureon(dim=10):
    cmd = ["python3", "aureon.py", "--objective", "rosenbrock", "--dimensions", str(dim), "--rgd_steps", "500"]
    start = time.time()
    output = check_output(cmd).decode("utf-8")
    end = time.time()
    return end - start, output

def run_scipy(dim=10):
    x0 = np.random.uniform(-2,2,dim)
    start = time.time()
    res = minimize(rosenbrock, x0, method='BFGS')
    end = time.time()
    return end - start, res

if __name__ == "__main__":
    aureon_time, aureon_output = run_aureon()
    scipy_time, scipy_res = run_scipy()
    print("Aureon Time:", aureon_time, "Final Loss:", "see output logs")
    print("SciPy BFGS Time:", scipy_time, "Final Loss:", scipy_res.fun)
    # This gives a rough idea of performance comparison.
