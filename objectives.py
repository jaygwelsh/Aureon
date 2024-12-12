import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

@njit(parallel=True, fastmath=True)
def rosenbrock(x):
    """
    Rosenbrock function.
    """
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

@njit(parallel=True, fastmath=True)
def rosenbrock_grad(x):
    """
    Gradient of the Rosenbrock function.
    """
    xm = x[1:]
    xm_m1 = x[:-1]
    grad = np.zeros_like(x)
    for i in prange(len(x)-1):
        grad[i+1] += 200*(xm[i]-xm_m1[i]**2)
        grad[i] += -400*xm_m1[i]*(xm[i]-xm_m1[i]**2)-2*(1-xm_m1[i])
    return grad

@njit(parallel=True, fastmath=True)
def rastrigin(x):
    """
    Rastrigin function.
    """
    n = x.shape[0]
    return 10*n + np.sum(x**2 - 10.0*np.cos(2*np.pi*x))

@njit(parallel=True, fastmath=True)
def rastrigin_grad(x):
    """
    Gradient of the Rastrigin function.
    """
    grad = np.zeros_like(x)
    for i in prange(x.shape[0]):
        grad[i] = 2*x[i] + 20*np.pi*np.sin(2*np.pi*x[i])
    return grad

def get_objective_and_gradient(name: str, dimensions: int):
    """
    Return the objective function and gradient based on the name.
    """
    if name == 'rosenbrock':
        return rosenbrock, rosenbrock_grad
    elif name == 'rastrigin':
        return rastrigin, rastrigin_grad
    else:
        raise ValueError(f"Unknown objective: {name}")

def get_available_functions():
    """
    Return a list of available objectives.
    """
    return ['rosenbrock', 'rastrigin']  # you can list more if implemented

class MultiObjectiveCombiner:
    def __init__(self, funcs_grads, weights):
        self.funcs_grads = funcs_grads
        self.weights = weights

    def func(self, x):
        val = 0.0
        for (f, _), w in zip(self.funcs_grads, self.weights):
            val += w * f(x)
        return val

    def grad(self, x):
        g = np.zeros_like(x)
        for (_, gfunc), w in zip(self.funcs_grads, self.weights):
            g += w * gfunc(x)
        return g

class ParetoManager:
    def __init__(self, funcs_grads):
        self.funcs = [fg[0] for fg in funcs_grads]
        self.solutions = []

    def add_solution(self, x):
        objs = [f(x) for f in self.funcs]
        self.solutions.append({'x': x.copy(), 'objs': objs})

    def compute_pareto_front(self):
        solutions = self.solutions
        if not solutions:
            return []
        n = len(solutions)
        S = [[] for _ in range(n)]
        n_dom = [0]*n

        for p in range(n):
            for q in range(n):
                if p != q:
                    if dominates(solutions[p]['objs'], solutions[q]['objs']):
                        S[p].append(q)
                    elif dominates(solutions[q]['objs'], solutions[p]['objs']):
                        n_dom[p] += 1

        fronts = [[]]
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)

        i = 0
        # Instead of 'while fronts[i]:', we ensure i is in range and front is non-empty
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                # No new front formed, break the loop.
                break

        # The first front in `fronts` is the Pareto front
        first_front = fronts[0] if fronts else []
        return [solutions[idx] for idx in first_front]


    def plot_pareto_front_2d(self):
        solutions = self.solutions
        xs = [sol['objs'][0] for sol in solutions]
        ys = [sol['objs'][1] for sol in solutions]
        plt.figure()
        plt.scatter(xs, ys, c='blue', label='All Solutions')
        front = self.compute_pareto_front()
        fxs = [sol['objs'][0] for sol in front]
        fys = [sol['objs'][1] for sol in front]
        plt.scatter(fxs, fys, c='red', label='Pareto Front')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front')
        plt.legend()
        plt.show()

    def plot_pareto_front_3d(self):
        from mpl_toolkits.mplot3d import Axes3D
        solutions = self.solutions
        xs = [sol['objs'][0] for sol in solutions]
        ys = [sol['objs'][1] for sol in solutions]
        zs = [sol['objs'][2] for sol in solutions]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c='blue', label='All Solutions')
        front = self.compute_pareto_front()
        fxs = [sol['objs'][0] for sol in front]
        fys = [sol['objs'][1] for sol in front]
        fzs = [sol['objs'][2] for sol in front]
        ax.scatter(fxs, fys, fzs, c='red', label='Pareto Front')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        plt.title('Pareto Front (3D)')
        plt.legend()
        plt.show()

def dominates(a, b):
    return all(a_i <= b_i for a_i, b_i in zip(a,b)) and any(a_i < b_i for a_i, b_i in zip(a,b))

def apply_log_barrier_method(func, lb, ub, penalty_scale):
    def barriered_func(x):
        val = func(x)
        if np.any(x <= lb) or np.any(x >= ub):
            violation = np.sum(np.maximum(lb - x, 0) + np.maximum(x - ub, 0))
            val += penalty_scale * violation**2
        else:
            margin_lb = x - lb
            margin_ub = ub - x
            val -= penalty_scale*(np.sum(np.log(margin_lb)) + np.sum(np.log(margin_ub)))
        return val
    return barriered_func