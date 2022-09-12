import random
import time

from tqdm import tqdm
import networkx as nx
import numpy as np


def sum_graph_weights(graph: nx.Graph):
    """
    Given a weighted graph the function returns the sum of edge weights in a graph.

    :param graph: Graph.

    :return: Sum of edge weights in the given graph.
    """
    sum_w = 0
    for source, target in graph.edges:
        sum_w += graph[source][target]["weight"]

    return sum_w


def get_weights_vector(graph: nx.Graph):
    """
    Given a weighted graph the function returns the vector of graph weights.

    :param graph: Graph.

    :return: Vector of graph weights.
    """
    w = []
    for source, target in graph.edges:
        w.append(graph[source][target]["weight"])

    return w


class MaximalWeightMatchingLocalSearch():
    """
    Class that implements the Local Search algorithm used to solve the maximal weight matching problem.
    """

    def __init__(self,
                 G: nx.Graph,
                 k: int,
                 max_iter=10000,
                 optimized_edge_picking=False):
        # Seed
        random.seed(0)
        # Initialize variables that are passed as arguments
        self.G = G
        self.k = k
        self.max_iter = max_iter
        self.optimized_edge_picking = optimized_edge_picking
        self.w = get_weights_vector(self.G)

        # Create the initial (empty) matching
        self.M = nx.Graph()
        self.M_sum_w = 0
        self.M_w = []

    def single_step(self):
        """
        Makes a single step of the local search.
        """
        # Randomly pick the number of edges that will be removed to the matching
        n_edges_to_remove = random.randint(0, self.k)
        # Randomly pick the number of edges that will be added to the matching
        n_edges_to_add = random.randint(0, self.k)
        # Initialize the next potential matching
        M_next = self.M.copy()
        M_next_w = self.M_w
        # Remove <=k edges
        for i in range(n_edges_to_remove):
            # If matching is not empty. In this case we cannot remove edges from an empty matching
            if len(M_next.edges) > 0:
                if self.optimized_edge_picking:
                    weights = list(map(lambda x: (2 - x) / sum(2 - np.array(M_next_w)), list(M_next_w)))
                    M_random_edge = random.choices(population=list(M_next.edges), weights=weights, k=1)[0]
                else:
                    M_random_edge = list(M_next.edges)[random.randint(0, len(M_next.edges) - 1)]

                M_next.remove_edge(M_random_edge[0], M_random_edge[1])
                if self.optimized_edge_picking:
                    M_next_w = get_weights_vector(graph=M_next)
        # Add <=k edges
        for i in range(n_edges_to_add):
            if self.optimized_edge_picking:
                weights = list(map(lambda x: (x - 1) / sum(np.array(self.w) - 1), list(self.w)))
                G_random_edge = random.choices(population=list(self.G.edges(data=True)), weights=weights, k=1)[0]
            else:
                # Sample a random edge
                G_random_edge = list(self.G.edges(data=True))[random.randint(0, len(self.G.edges) - 1)]

            # Check if matching constrain of vertex incidence will persist if adding the sampled edge
            if not M_next.has_node(G_random_edge[0]) and not M_next.has_node(G_random_edge[1]):
                M_next.add_node(G_random_edge[0])
                M_next.add_node(G_random_edge[1])
                M_next.add_edge(G_random_edge[0], G_random_edge[1], weight=G_random_edge[2]['weight'])


        # Calculate the sum of weights in the next potential matching and replace the old matching with the new
        # matching if the sum of weights is lower (thus, solution is improved).
        M_next_sum_w = sum_graph_weights(M_next)
        if self.M_sum_w < M_next_sum_w:
            self.M = M_next
            self.M_sum_w = M_next_sum_w
            self.M_w = get_weights_vector(self.M)

    def optimize(self):
        """
        Optimizes the maximal weight matching problem of a given graph G.
        """
        start_time = time.time()

        step = 0
        for i in tqdm(range(self.max_iter)):
            # Make a single step of local search
            self.single_step()
            step += 1

        total_time = time.time() - start_time

        # Print results
        print(f"Sum of weights={self.M_sum_w}")
        print(f"Time spent: {total_time}, Steps: {step}")
