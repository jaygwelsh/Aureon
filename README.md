
# Aureon

**Aureon** is a powerful hybrid optimization tool that integrates **Simulated Annealing (SA)** for global exploration with **Resilient Gradient Descent (RGD)** for local refinement. Designed to tackle complex optimization problems, Aureon supports **multi-objective optimization**, **robust constraint handling**, and leverages **performance enhancements** like **JIT compilation** and **parallel processing** to deliver accurate and efficient results.

## Features

- **Hybrid Optimization**: Combines global search (SA) with local refinement (RGD) for robust optimization.
- **Multi-Objective Optimization**: Optimize multiple conflicting objectives with Pareto front generation.
- **Constraint Handling**: Implements log-barrier methods for effective constraint management.
- **Performance Enhancements**:
  - **JIT Compilation**: Utilizes Numba for accelerated function evaluations.
  - **Vectorization**: Employs NumPy's vectorized operations for efficient computations.
  - **Parallel Processing**: Leverages parallel loops via Numba to speed up gradient calculations.
- **Custom Objectives**: Easily integrate custom objective functions and gradients.
- **Benchmarking**: Compare Aureon's performance against industry-standard optimizers like SciPy's BFGS.
- **Educational Tool**: Transparent and modular codebase ideal for learning optimization algorithms.

## Installation

Ensure you have **Python 3.7+** installed. It's recommended to use a virtual environment.

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/aureon.git
    cd aureon
    ```

2. **Create and Activate a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scriptsctivate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Single Objective Optimization

Optimize the **Rosenbrock** function in a 10-dimensional space:
```bash
python3 aureon.py --objective rosenbrock --dimensions 10
```

### Multi-Objective Optimization

Optimize both **Rosenbrock** and **Rastrigin** functions with equal weights in a 3-dimensional space:
```bash
python3 aureon.py --multi_objective rosenbrock rastrigin --weights 0.5 0.5 --dimensions 3
```

### Constraint Handling

Optimize the **Rosenbrock** function within the bounds \([-1, 2]\) for each dimension:
```bash
python3 aureon.py --objective rosenbrock --dimensions 5 --constraint_bounds -1 2 --constraint_penalty 1000
```

### Custom Objective Functions

Define a custom objective by creating a `custom_objective.py`:
```python
# custom_objective.py

import numpy as np
from numba import njit, parallel

@njit(parallel=True, fastmath=True)
def custom_objective(x):
    # Example: Sphere function
    return np.sum(x**2)

@njit(parallel=True, fastmath=True)
def custom_gradient(x):
    return 2 * x
```

Run Aureon with your custom objective:
```bash
python3 aureon.py --objective custom --objective_file custom_objective.py --dimensions 3
```

## Benchmarking

Compare Aureon's performance against SciPy's BFGS optimizer.

Run the benchmark script:
```bash
python3 benchmarks/benchmark.py
```

**Sample Results:**

| **Optimizer**  | **Time (s)** | **Final Loss**        |
|----------------|--------------|-----------------------|
| **Aureon**     | 1.71         | 0.000000              |
| **SciPy BFGS** | 0.013         | 6.0282e-11            |

*Note: Aureon achieves near-zero loss, comparable to SciPy's highly optimized BFGS optimizer, demonstrating its effectiveness.*

## Testing

Ensure all functionalities work as expected using the test suite.

Run tests with `pytest`:
```bash
pytest
```

## Integration

Aureon can be integrated with other scientific libraries to enhance its capabilities.

### Example: Using Aureon with pymoo

```python
from pymoo.factory import get_problem, get_algorithm
from pymoo.optimize import minimize
import numpy as np

# Example integration logic
initial_points = np.array([...])  # Replace with actual initial points from Aureon
algorithm = get_algorithm("ga")    # Genetic Algorithm from pymoo

result = minimize(get_problem("rosenbrock"),
                  algorithm,
                  termination=('n_gen', 100),
                  seed=1,
                  verbose=True)
```

## Documentation and Examples

Comprehensive documentation and examples are available to help you get started.

- **Docstrings**: Detailed explanations within the codebase.
- **Tutorials**: Located in the `examples/` directory, including Jupyter notebooks for:
  - Basic and multi-objective optimizations
  - Constraint applications
  - Custom objectives
  - Benchmarking comparisons
- **Visualization**: Utilize Matplotlib for loss history and Pareto front plots.

**Example: Plotting Loss History**
```python
import matplotlib.pyplot as plt

# Assuming rgd_result contains 'loss_history'
plt.plot(rgd_result['loss_history'], label='RGD Loss')
plt.title('Aureon: RGD Loss History')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
```

## Contributing

Contributions are welcome! Whether it's improving performance, adding new features, or enhancing documentation, your input can help make Aureon better.

### How to Contribute

1. **Fork the Repository**
2. **Clone Your Fork**
    ```bash
    git clone https://github.com/yourusername/aureon.git
    cd aureon
    ```
3. **Create a New Branch**
    ```bash
    git checkout -b feature/YourFeatureName
    ```
4. **Make Your Changes**
5. **Commit Your Changes**
    ```bash
    git commit -m "Add feature XYZ"
    ```
6. **Push to Your Fork**
    ```bash
    git push origin feature/YourFeatureName
    ```
7. **Create a Pull Request** on the original repository.

### Coding Standards

- Follow PEP 8 guidelines.
- Include docstrings and comments for new features.
- Write unit tests for new functionalities.

## License

This project is licensed under the [MIT License](LICENSE).