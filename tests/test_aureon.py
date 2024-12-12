import pytest
from subprocess import check_output

def test_basic_run():
    cmd = ["python3", "aureon.py", "--dimensions", "5"]
    output = check_output(cmd).decode("utf-8")
    assert "Simulated Annealing:" in output
    assert "Resilient Gradient Descent:" in output

def test_custom_objective_run():
    # Ensure a recognizable stdout message is printed.
    cmd = ["python3", "aureon.py", "--objective", "custom", "--objective_file", "custom_objective.py", "--dimensions", "3"]
    output = check_output(cmd).decode("utf-8")
    assert "Using custom objective" in output

def test_multi_objective_run():
    cmd = ["python3", "aureon.py", "--multi_objective", "rosenbrock", "rastrigin", "--weights", "0.5", "0.5", "--dimensions", "3"]
    output = check_output(cmd).decode("utf-8")
    assert "Using multi-objective mode" in output
