import time
from typing import Callable
import numpy as np


class NelderMeadOptimizer():
    """
    Class that implements the Nelder Mead method optimization and returns the optimal solution of the given function.
    """
    def __init__(self,
                 x_origin: list[float],
                 diameter: float,
                 f_cost: Callable,
                 x_true: list[float] = None,
                 max_iter=1000,
                 eps=1e-6):
        # Cost function init
        self.f_cost = f_cost
        # Max iteation init - one of the stopping criteria
        self.max_iter = max_iter
        # Epsilon init - one of the stopping criteria
        self.eps = eps
        # Ground truth init
        self.x_true = x_true

        # Calculate dimension
        self.n = len(x_origin)
        # Initializes the sample points. In continuation this vector is continously updated.
        self.x = self.generate_x_from_origin_with_diameter(x_origin, diameter)
        # Calculate vector y
        self.y = np.array(list(map(lambda x_i: self.f_cost(x_i), self.x)))

        # Fix x and y so that y_0 <= y_1 <= ... <= y_n
        self.rearrange_xy()

    def rearrange_xy(self):
        """
        Rearranges x and y vectors so that they satisfy:
        y_0 <= y_1 <= ... <= y_n
        and
        for all i, x_i corresponds to y_i
        """
        # Sorts function values and returns indices so that we can rearrange x and y vectors
        sorted_indices = np.argsort(self.y)
        self.x = self.x[sorted_indices]
        self.y = self.y[sorted_indices]

    def reflect(self, x_mean):
        """
        Returns reflection point of worst point.

        :param x_mean: Arithmetic mean of points x_0, ... x_n-1

        :return: Reflection point.
        """
        return x_mean + 1 * (x_mean - self.x[self.n])

    def expand(self, x_mean):
        """
        Returns expansion point of worst point.

        :param x_mean: Arithmetic mean of points x_0, ... x_n-1

        :return: Expansion point.
        """
        return x_mean + 2 * (x_mean - self.x[self.n])

    def contract_outside(self, x_mean):
        """
        Returns outside contraction point of worst point.

        :param x_mean: Arithmetic mean of points x_0, ... x_n-1

        :return: Outside contraction point.
        """
        return x_mean + 0.5 * (x_mean - self.x[self.n])

    def contract_inside(self, x_mean):
        """
        Returns inside contraction point of worst point.

        :param x_mean: Arithmetic mean of points x_0, ... x_n-1

        :return: Inside contraction point.
        """
        return x_mean - 0.5 * (x_mean - self.x[self.n])

    def shrink(self):
        """
        For all i > 0, sets x_i as arithmetic mean of x_i and x_0 -> x_i = (x_0 + x_i) / 2
        """
        # + 1 to include n
        for i in range(1, self.n + 1):
            self.x[i] = 0.5 * (self.x[0] + self.x[i])

    def single_step(self):
        """
        Makes a single step of the Nelder Mead method.
        """
        x_mean = sum(self.x[:(self.n), :]) / self.n

        x_r = self.reflect(x_mean)
        y_r = self.f_cost(x_r)

        if y_r < self.y[0]:
            # Try to expand
            x_e = self.expand(x_mean)
            y_e = self.f_cost(x_e)
            if y_e < y_r:
                # Expand
                self.x[self.n] = x_e
            else:
                # Reflect
                self.x[self.n] = x_r

        elif y_r < self.y[self.n - 1]:
            # Reflect
            self.x[self.n] = x_r
        elif y_r < self.y[self.n]:
            # Try to contract outside
            x_co = self.contract_outside(x_mean)
            y_co = self.f_cost(x_co)
            if y_co < y_r:
                # Contract outside
                self.x[self.n] = x_co
            else:
                # Shrink
                self.shrink()
        else:
            # Try to contract inside
            x_ci = self.contract_inside(x_mean)
            y_ci = self.f_cost(x_ci)
            if y_ci < y_r:
                # Contract inside
                self.x[self.n] = x_ci
            else:
                # Shrink
                self.shrink()

        # Recalculate vector y
        self.y = np.array(list(map(lambda x_i: self.f_cost(x_i), self.x)))

        # Rearrange x and y vector
        self.rearrange_xy()

    def is_termination_criteria_met(self, step):
        """
        Given the current step we check if the termination criteria is met.

        :param step: Current step of Nelder Mead method.

        :return: Boolean if termination criteria is met.
        """
        # We surpassed the maximum number of iterations -> This way we avoid an infinite loop
        if step == self.max_iter:
            return True
        # We are sufficiently close in function values
        if (self.y[self.n] - self.y[0]) < self.eps:
            return True

        return False

    def minimize(self):
        """
        Minimize function runs the Nelder Mead method. Until function values are not sufficiently close it
        keeps improving the points with reflection, expansion, contraction and shrinkage.

        :return: Returns optimal sample points and corresponding function values.
        """
        start_time = time.time()

        step = 0
        while not self.is_termination_criteria_met(step=step):
            self.single_step()
            step += 1
        solution = self.x[0]

        total_time = time.time() - start_time

        print(f"Solution: x={solution}, y={self.f_cost(solution)}")
        if self.x_true is not None:
            print(f"Abs. diff. to true x: {np.abs(solution - self.x_true)}")
            print(f"Abs. diff. to true y: {abs(self.f_cost(solution) -self.f_cost(self.x_true))}")
        print(f"Time spent: {total_time}, Steps: {step}")
        return self.x

    def generate_x_from_origin_with_diameter(self, x_origin, diameter):
        # Set initial distances to 1 and 0 on diagonal
        tetrahedron = np.ones([self.n, self.n]) - np.diag(np.ones(self.n))
        tetrahedron = tetrahedron / np.sqrt(self.n - 1) * diameter

        return np.vstack([x_origin, x_origin + tetrahedron])
