import time
from tqdm import tqdm

import numpy as np


class InteriorPointMethodMinimizer():
    """
    Class that implements the Interior Point Method minimization and returns the optimal solution of the system.
    """

    def __init__(self,
                 A: np.array,
                 b: np.array,
                 c: np.array,
                 norm_integer: int = 1000,
                 max_iter: int = 1000,
                 forced_delta: float = None,
                 verbose: int = 0):
        """
        Initialization of the interior point method.

        :param A: Constraint coefficient matrix with (ideally) linearly independent rows.
        :param b: Upper bound vector of constraints.
        :param c: Cost vector.
        :param norm_integer: To avoid numbers that are too big for Python to handle we norm the matrices by some, big enough integer.
        :param max_iter: Maximum number of iterations to avoid infinite looping.
        :param forced_delta: Forced delta that replaces the calculated delta if given as a function parameter. This is solely for test purposes.
        """
        # Initialize the input parameters as class attributes
        self.A = A
        self.b = b
        self.c = c
        self.max_iter = max_iter
        self.forced_delta = forced_delta
        self.verbose = verbose

        # Number of rows - number of constraints
        self.m = self.A.shape[0]
        # Number of columns - number of variables
        self.n = self.A.shape[1]

        # In theory we would need to adapt matrix A so that it has a full rank
        # Solving the given cases, this is not really needed
        if np.linalg.matrix_rank(self.A) < self.m and self.verbose > 0:
            print("\nMatrix A does not have full rank\n")

        # Define variable that help derive the initial solution:

        # Primal problem (P')
        # min c.T @ x + M * x_(n+2) subject to
        #         (1) A @ x + rho @ x_(n+2) = d
        #         (2) e.T @ x + x_(n+1) + x_(n+1) = n + 2
        #
        # Dual problem (D')
        # max d.T @ y + (n + 2) * y_(m+1)
        #         (1) A.T @ y + e.T

        # U >= a_ij, b_i, c_j for all i_j
        self.U = max(np.max(np.abs(self.A / norm_integer)), np.max(np.abs(self.b / norm_integer)),
                     np.max(np.abs(self.c / norm_integer)))
        # If the original problem is feasible then there is a feasible solution such that all coordinates are
        # bounded by W. Additionally if problem is bounded than this solution is optimal.
        self.W = (self.m * self.U) ** self.m
        # Derived from (P)(1)
        self.d = ((self.b / norm_integer) / self.W).reshape(-1, 1)
        # Simple unit vector
        self.e = np.ones(shape=(self.n, 1))
        # Derived from (P)(1)
        self.rho = self.d - (self.A / norm_integer) @ self.e
        # Derivation from Claim 6
        self.delta = 1 / (8 * np.sqrt(self.n + 2))

        # Choice of M
        self.R = 1 / ((self.W ** 2) * (2 * self.n * ((self.m + 1) * self.U) ** (3 * (self.m + 1))))
        self.M = 4 * self.n * self.U / self.R

        # Follows from (P') - left-hand side
        self.A_prime = np.block([
            [A, np.zeros(shape=(self.m, 1)), self.rho],
            [self.e.T, 1, 1],
        ])
        # Right-hand side of the (P')
        self.b_prime = np.block([
            [self.d],
            [self.n + 2]
        ])
        # (n + 2)-dim vector of ones
        self.e_prime = np.ones(shape=(self.A_prime.shape[1], 1))

        # Set the initial feasible solution x_0, y_0, s_0, mu_0
        # Set initial mu_0
        # Follows from calculation of sigma^2 (deviation of x_i * s_i)
        self.mu_i = 2 * (np.square(self.M) + np.sum(
            np.square(self.c))) ** 0.5  # 2 * np.sqrt(M ** 2 + np.sum(c ** 2))


        self.c = self.c.reshape(-1, 1)

        # Set initial x_0
        self.x_i = np.ones((self.n + 2, 1))
        self.x_i = self.x_i.reshape(-1, 1)

        # Set initial y_0
        self.y_i = np.zeros(self.m + 1)
        # Set y_i[m+1] = -mu_i
        self.y_i[self.m] = -self.mu_i
        self.y_i = self.y_i.reshape(-1, 1)

        # Set initial s_0
        self.s_i = np.block([
            [self.c + self.e * self.mu_i],
            [self.mu_i],
            [self.M + self.mu_i]
        ])

    def iterative_improvement(self):
        """
        Make a single step of the iterative improvement process.
        Solving system (S), where:
            A @ h=0
            A.T @ k + f = 0
            x_i * f_i + s_i * h_i = mu_i - x_i * s_i
        """
        # Construct diagonal matrices X, S and the inverse of S
        X = np.diag(self.x_i.reshape(-1))
        S = np.diag(self.s_i.reshape(-1))
        S_inv = np.linalg.inv(S)

        # Calculate the steps for x, s and y that will be used to calculate x', s' and y'
        # S_inv @ X - if we were to sum up the diagonal elements we would get the objective gap
        # between primal and dual problem.
        k = np.dot(
            np.linalg.pinv(self.A_prime @ S_inv @ X @ self.A_prime.T),
            self.b_prime - np.dot((self.mu_i * self.A_prime) @ S_inv, self.e_prime)
        )
        f = - self.A_prime.T @ k
        h = - X @ S_inv @ f + (self.mu_i * S_inv) @ self.e_prime - self.x_i

        # Calculate x', s', y' and mu'
        self.x_i = self.x_i + h
        self.y_i = self.y_i + k
        self.s_i = self.s_i + f

        # If delta is given as the parameter, overwrite the calculated delta.
        delta = self.forced_delta if self.forced_delta is not None else (1 - self.delta)
        self.mu_i = delta * self.mu_i

    def round_solution(self, Q: float):
        """
        The final rounding procedure calculates the optimal solution x, given the final x, y and s vectors.

        :param Q: Matrix Q that satisfies condition:
            either
                x_i_optimal >= Q and s_i_optimal = 0
                or
                s_i_optimal >= Q and x_i_optimal = 0
        :return: Optimal solution x to the given system.
        """
        # Initialize variable where we will store optimal x and s
        x_optimal = self.x_i
        s_optimal = self.s_i

        # Where values of x and s are lower than the sensitivity, we set the values to 0
        x_optimal[x_optimal < Q / (4 * (self.n + 2))] = 0
        s_optimal[s_optimal < Q / (4 * (self.n + 2))] = 0

        # Set B - index set where slack variables are 0
        B = list(filter(lambda x: x < self.n, np.argwhere(s_optimal.flatten() == 0).flatten()))
        # Set N - index set where slack variables are 0
        N = list(filter(lambda x: x < self.n, np.argwhere(x_optimal.flatten() == 0).flatten()))

        # Solve the system of A_B @ x_B_optimal = b and set solution to optimal x at indices B.
        A_B_inv = np.linalg.pinv(self.A[:, B])
        x_optimal[B] = np.dot(A_B_inv, self.b).reshape(-1, 1)

        # Return optimal x without the additional variables that were needed in the method.
        return x_optimal[:self.n]

    def optimize(self):
        start_time = time.time()

        # Calculate the theoretical minimum mu: mu_f
        Q = self.R / (self.n + 2)
        mu_f = (self.R * Q) / (64 * ((self.n + 2) ** 2) * ((self.m + 1) * self.U) ** (self.m + 2))
        for _ in range(self.max_iter):
            # Check if interior point method has converged to the theoretical minimum that it can achieve
            if self.mu_i < mu_f:
                if self.verbose > 0:
                    print("Interior point method converged")
                break

            # Make a single step of iterative improvement
            self.iterative_improvement()

        # Print optimal solution x and time spent in execution
        x_optimal = self.round_solution(Q=Q)
        total_time = time.time() - start_time
        if self.verbose > 0:
            print(f"x: {x_optimal}")
            print(f"Time spent: {total_time}")

        # Return optimal solution
        return x_optimal
