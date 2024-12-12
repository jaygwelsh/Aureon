#!/usr/bin/env python3
import argparse
import sys
import logging
from typing import Callable, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
import importlib.util
import os

from objectives import get_objective_and_gradient, MultiObjectiveCombiner, ParetoManager, get_available_functions, apply_log_barrier_method
# Removed the complicated partial-chunk gradient code
# Now relying on Numba parallel in objective gradients themselves

class ResilientGradientDescent:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_steps: int = 500,
        learning_rate: float = 0.1,
        decay_rate: float = 0.8,
        patience: int = 50,
        beta: float = 0.9,
        max_gradient: float = 0.5,
        tolerance: float = 1e-6
    ) -> None:
        self.func = func
        self.grad = grad
        self.x = np.array(x0, dtype=float)
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.patience = patience
        self.beta = beta
        self.max_gradient = max_gradient
        self.tolerance = tolerance

        self.loss_history: List[float] = []
        self.learning_rate_history: List[float] = []
        self.no_improve_steps = 0
        self.velocity = np.zeros_like(self.x)
        self.best_loss = float('inf')
        self.best_x = self.x.copy()

    def optimize(self) -> Dict[str, Any]:
        for step in range(1, self.max_steps + 1):
            gradient = self.grad(self.x)
            gradient = np.clip(gradient, -self.max_gradient, self.max_gradient)
            self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
            self.x -= self.learning_rate * self.velocity
            loss = self.func(self.x)
            self.loss_history.append(loss)
            self.learning_rate_history.append(self.learning_rate)

            logging.debug(
                f"RGD Step {step}: Loss={loss:.8f}, LR={self.learning_rate:.6f}, Grad={gradient}, x={self.x}"
            )

            if loss < self.best_loss - self.tolerance:
                self.best_loss = loss
                self.best_x = self.x.copy()
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1

            if self.no_improve_steps >= self.patience:
                old_lr = self.learning_rate
                self.learning_rate *= self.decay_rate
                self.no_improve_steps = 0
                logging.info(
                    f"Decaying learning rate from {old_lr:.6f} to {self.learning_rate:.6f} "
                    f"at step {step} due to lack of improvement."
                )

            if loss <= self.tolerance:
                logging.info(f"Converged at step {step} with loss {loss:.8f}")
                break

        return {
            'final_loss': self.best_loss,
            'optimal_point': self.best_x,
            'loss_history': self.loss_history,
            'learning_rate_history': self.learning_rate_history,
            'total_steps': step,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'decay_rate': self.decay_rate,
            'beta': self.beta,
            'max_gradient': self.max_gradient,
            'tolerance': self.tolerance
        }

def load_custom_objective(file_path: str):
    if not os.path.exists(file_path):
        logging.error(f"Custom objective file '{file_path}' not found.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    if not hasattr(custom_module, "custom_objective") or not hasattr(custom_module, "custom_gradient"):
        logging.error("Custom objective file must define `custom_objective` and `custom_gradient`.")
        sys.exit(1)

    return custom_module.custom_objective, custom_module.custom_gradient

def run_aureon(
    objective: str,
    dimensions: int,
    sa_steps: int,
    rgd_steps: int,
    learning_rate: float,
    decay_rate: float,
    patience: int,
    beta: float,
    max_gradient: float,
    tolerance: float,
    visualize: bool,
    log_level: str,
    objective_file: str,
    multi_objective: List[str],
    weights: List[float],
    constraint_bounds: List[float],
    constraint_penalty: float
) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f'Invalid log level: {log_level}')
        sys.exit(1)
    logging.basicConfig(level=numeric_level, format='%(levelname)s: %(message)s')

    if multi_objective and objective == 'custom':
        logging.error("Multi-objective with custom objective not supported simultaneously in this demo.")
        sys.exit(1)

    if multi_objective:
        funcs_grads = []
        for obj_name in multi_objective:
            f, g = get_objective_and_gradient(obj_name, dimensions)
            funcs_grads.append((f,g))
        combined = MultiObjectiveCombiner(funcs_grads, weights)
        func, grad = combined.func, combined.grad
        print("Using multi-objective mode (weighted sum + Pareto approximation)")
        logging.info(f"Using multi-objective mode with {multi_objective} and weights {weights}")
        pareto_manager = ParetoManager(funcs_grads)
    elif objective == 'custom':
        if not objective_file:
            logging.error("Custom objective selected but no file provided.")
            sys.exit(1)
        func, grad = load_custom_objective(objective_file)
        print("Using custom objective")
        logging.info(f"Using custom objective from {objective_file}")
        pareto_manager = None
    else:
        func, grad = get_objective_and_gradient(objective, dimensions)
        logging.info(f"Using built-in objective: {objective}")
        pareto_manager = None

    # Use log-barrier for constraints if provided
    if constraint_bounds:
        if len(constraint_bounds) != 2:
            logging.error("Constraint bounds must be two values: lb and ub.")
            sys.exit(1)
        lb, ub = constraint_bounds
        func = apply_log_barrier_method(func, lb, ub, constraint_penalty)

    x0 = np.random.uniform(-2, 2, size=dimensions)
    logging.info(f"Initial Random Point: {x0}")

    logging.info("\n--- Optimizing Objective with Aureon ---\n")
    logging.info("--- Simulated Annealing ---")

    sa_result = dual_annealing(
        func,
        bounds=[(-2, 2)] * dimensions,
        maxiter=sa_steps
    )
    sa_final_loss = sa_result.fun
    sa_optimal_point = sa_result.x

    logging.info(f"SA Final Loss: {sa_final_loss:.6f}")
    logging.info(f"SA Optimal Point: {sa_optimal_point}")

    logging.info("\n--- Resilient Gradient Descent ---")
    rgd_optimizer = ResilientGradientDescent(
        func=func,
        grad=grad,
        x0=sa_optimal_point,
        max_steps=rgd_steps,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        patience=patience,
        beta=beta,
        max_gradient=max_gradient,
        tolerance=tolerance
    )
    rgd_result = rgd_optimizer.optimize()

    rgd_final_loss = rgd_result['final_loss']
    rgd_optimal_point = rgd_result['optimal_point']

    logging.info(f"RGD Final Loss: {rgd_final_loss:.6f}")
    logging.info(f"RGD Optimal Point: {rgd_optimal_point}")

    logging.info("\n--- RGD Detailed Summary ---")
    logging.info(f"Initial Point (from SA): {sa_optimal_point}")
    logging.info(f"Final Point: {rgd_optimal_point}")
    logging.info(f"Final Loss: {rgd_final_loss:.6f}")
    logging.info(f"Total Steps: {rgd_result['total_steps']}")
    logging.info(f"Learning Rate History: {rgd_result['learning_rate_history']}")
    logging.info(f"Patience: {rgd_result['patience']}")
    logging.info(f"Decay Rate: {rgd_result['decay_rate']}")
    logging.info(f"Momentum Factor (beta): {rgd_result['beta']}")
    logging.info(f"Max Gradient: {rgd_result['max_gradient']}")
    logging.info(f"Tolerance: {rgd_result['tolerance']}")

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.plot(rgd_result['loss_history'], label='RGD Loss')
        plt.title('Aureon: RGD Loss History')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    if multi_objective:
        # Store final solution and compute Pareto front
        pareto_manager.add_solution(rgd_optimal_point)
        pareto_solutions = pareto_manager.compute_pareto_front()
        logging.info("Pareto Front Solutions:")
        for sol in pareto_solutions:
            logging.info(f"Point: {sol['x']} Objectives: {sol['objs']}")
        # Visualize if 2 or 3 objectives
        if len(multi_objective) == 2:
            pareto_manager.plot_pareto_front_2d()
        elif len(multi_objective) == 3:
            pareto_manager.plot_pareto_front_3d()

    print("\n--- Optimizing Objective with Aureon ---\n")
    print("--- Hybrid Optimization Summary ---\n")
    print("Simulated Annealing:")
    print(f"  Final Loss: {sa_final_loss:.6f}")
    print(f"  Optimal Point: {sa_optimal_point}\n")
    print("Resilient Gradient Descent:")
    print(f"  Final Loss: {rgd_final_loss:.6f}")
    print(f"  Optimal Point: {rgd_optimal_point}\n")
    print("RGD Detailed Summary:")
    print(f"  Initial Point: {sa_optimal_point}")
    print(f"  Final Point: {rgd_optimal_point}")
    print(f"  Final Loss: {rgd_final_loss:.6f}")
    print(f"  Total Steps: {rgd_result['total_steps']}")
    print(f"  Learning Rate History: {rgd_result['learning_rate_history']}")
    print(f"  Patience: {rgd_result['patience']}")
    print(f"  Decay Rate: {rgd_result['decay_rate']}")
    print(f"  Momentum Factor (beta): {rgd_result['beta']}")
    print(f"  Max Gradient: {rgd_result['max_gradient']}")
    print(f"  Tolerance: {rgd_result['tolerance']}")

def main():
    parser = argparse.ArgumentParser(description="Aureon with improved parallelism, constraint handling, and multi-objective depth.")
    parser.add_argument('--objective', type=str, default='rosenbrock')
    parser.add_argument('--objective_file', type=str, default='')
    parser.add_argument('--dimensions', type=int, default=10)
    parser.add_argument('--sa_steps', type=int, default=1000)
    parser.add_argument('--rgd_steps', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--max_gradient', type=float, default=0.5)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--multi_objective', nargs='*', default=[])
    parser.add_argument('--weights', nargs='*', type=float, default=[])
    parser.add_argument('--constraint_bounds', nargs=2, type=float, default=[])
    parser.add_argument('--constraint_penalty', type=float, default=1000.0)

    args = parser.parse_args()

    if args.dimensions <= 0:
        print("Error: dimensions must be positive.")
        sys.exit(1)
    if args.sa_steps <= 0:
        print("Error: sa_steps must be positive.")
        sys.exit(1)
    if args.rgd_steps <= 0:
        print("Error: rgd_steps must be positive.")
        sys.exit(1)
    if args.learning_rate <= 0:
        print("Error: learning_rate must be positive.")
        sys.exit(1)
    if not (0 < args.beta < 1):
        print("Error: beta must be between 0 and 1.")
        sys.exit(1)
    if args.max_gradient <= 0:
        print("Error: max_gradient must be positive.")
        sys.exit(1)
    if args.tolerance <= 0:
        print("Error: tolerance must be positive.")
        sys.exit(1)
    if args.multi_objective and (len(args.multi_objective) != len(args.weights)):
        print("Error: The number of weights must match the number of objectives in multi-objective mode.")
        sys.exit(1)

    run_aureon(
        objective=args.objective,
        dimensions=args.dimensions,
        sa_steps=args.sa_steps,
        rgd_steps=args.rgd_steps,
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        patience=args.patience,
        beta=args.beta,
        max_gradient=args.max_gradient,
        tolerance=args.tolerance,
        visualize=args.visualize,
        log_level=args.log_level,
        objective_file=args.objective_file,
        multi_objective=args.multi_objective,
        weights=args.weights,
        constraint_bounds=args.constraint_bounds,
        constraint_penalty=args.constraint_penalty
    )

if __name__ == "__main__":
    main()
