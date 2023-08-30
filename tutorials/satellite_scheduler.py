# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import matplotlib.pyplot as plt
import networkx as ntx
import numpy as np

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from networkx.algorithms.approximation import maximum_independent_set
from typing import Tuple, Optional

from lava.utils import loihi

from lava.lib.optimization.problems.problems import QUBO
from lava.lib.optimization.solvers.generic.solver import OptimizationSolver, SolverConfig
from lava.lib.optimization.utils.generators.mis import MISProblem


def is_visible(satellite_y, request_y, view_height):
    """ Return whether the request is visible to the satellite. """
    return satellite_y <= request_y <= satellite_y + view_height

def is_same_satellite(n1, n2):
    """ Return whether nodes n1 and n2 reference the same satellite. """
    return n1[1] == n2[1]

def is_movable(n1, n2, turn_rate):
    """ Return whether a satellite can turn from n1 to n2 without
    exceeding turn_rate. """
    xdist = abs(n1[2] - n2[2])
    ydist = abs(n1[3] - n2[3])
    return turn_rate * xdist >= ydist

def is_same_request(n1, n2):
    """ Return whether n1 and n2 reference the same request. """
    return (n1[2] == n2[2]) and (n1[3] == n2[3])

def is_feasible(n1, n2, turn_rate):
    """ Return whether it is feasible to traverse from n1 to n2. """
    return not is_same_request(n1, n2) and \
        (not is_same_satellite(n1, n2) or is_movable(n1, n2, turn_rate))

def is_same_node(n1, n2):
    """ Return whether n1 and n2 are the same node. """
    return n1[0] == n2[0]


class SatelliteScheduleProblem:
    """
    SatelliteScheduleProblem is a synthetic scheduling problem in which a number of
    vehicles must be assigned to view as many requests in a 2-dimensional plane as
    possible. Each vehicle moves horizontally across the plane, has minimum and
    maximum view angle, and has a maximum rotation rate (i.e. the rate at which the
    vehicle can reorient vertically from one target to the next).

    The problem is represented as an infeasibility graph and can be solved by finding
    the Maximum Independent Set.
    """

    def __init__(
            self,
            num_satellites : int = 6,
            view_height : float = 0.25,
            view_coords : Optional[np.ndarray] = None,
            num_requests : int = 48,
            turn_rate : float = 2,
            qubo_weights : Tuple = (1, 8),
            solution_criteria : float = 0.99,
    ):
        """ Create a SatelliteScheduleProblem.
        Parameters
        ----------
        num_satellites : int, default = 6
            The number of satellites to generate schedules for.
        view_height : float, default = 0.25
            The range from minimum to maximum viewable angle for each satellite.
        view_coords : Optional[np.ndarray], default = None
            The view coordinates (i.e. minimum viewable angle) for each satellite
            in a numpy array. If None, view coordinates will be evenly distributed
            across the viewable range.
        num_requests : int, default = 48
            The number of requests to generate.
        turn_rate : float, default = 2
            How quickly each satellite may reorient its view angle.
        qubo_weights : Tuple, default = (1, 8)
            The QUBO weight matrix parameters for diagonal and off-diagonal weights.
        solution_criteria : float, default = 0.99
            The target for a successful solution. The solver will stop looking for
            a better schedule if the specified fraction of requests are satisfied.
        """
        super().__init__()
        self.num_satellites = num_satellites
        self.view_height = view_height
        if view_coords is None:
            self.view_coords = np.linspace(0 - view_height / 2,
                                           1 - view_height / 2,
                                           num_satellites)
        else:
            self.view_coords = view_coords
        self.num_requests = num_requests
        self.turn_rate = turn_rate
        self.qubo_weights = qubo_weights
        if type(solution_criteria) is float and 0.0 < solution_criteria <= 1.0:
            self.target_cost = int(-solution_criteria * num_requests * qubo_weights[0])
        elif type(solution_criteria) is int and solution_criteria < 0:
            self.target_cost = solution_criteria
        self.random_seed = None
        self.graph = None
        self.adjacency = None
        self.satellites = None
        self.requests = None
        self.netx_solution = None
        self.lava_backend = None
        self.probe_cost = None
        self.qubo_problem = None
        self.solver_report = None
        self.lava_solution = None
    
    def generate(self, seed=None):
        """ Generate a new scheduler problem. """
        if seed:
            self.random_seed = seed
            np.random.seed(seed)
        self.graph = ntx.Graph()
        self.satellites = range(self.num_satellites)
        self.generate_requests()
        self.generate_visible_nodes()
        self.generate_infeasibility_graph()
        self.rescale_adjacency()
    
    def generate_requests(self):
        """ Generate a random set of requests in the 2D plane. """
        self.requests = np.random.random((self.num_requests, 2))
        order = np.argsort(self.requests[:,0])
        self.requests = self.requests[order,:]

    def generate_visible_nodes(self):
        """
        Add nodes to the graph for every combination of a vehicle and a request visible
        to that vehicle.
        """
        node_id = 0
        for i in self.satellites:
            for j in range(self.num_requests):
                if is_visible(self.view_coords[i], self.requests[j,1], self.view_height):
                    self.graph.add_node((node_id, i, self.requests[j,0], self.requests[j,1]))
                    node_id += 1
        self.num_nodes = node_id
    
    def generate_infeasibility_graph(self):
        """
        Create edges between any two nodes of the graph which cannot be traversed and
        add corresponding weights to the adjacency matrix.
        """
        self.adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                if not is_same_node(n1, n2) and not is_feasible(n1, n2, self.turn_rate):
                    self.adjacency[n1[0], n2[0]] = 1
                    self.graph.add_edge(n1, n2)
    
    def rescale_adjacency(self):
        """ Scale the adjacency matrix weights for QUBO solver. """
        self.adjacency = np.triu(self.adjacency)
        self.adjacency += self.adjacency.T - 2 * np.diag(self.adjacency.diagonal())

    def solve_with_netx(self):
        """ Find an approximate maximum independent set using networkx. """
        solution = maximum_independent_set(self.graph)
        self.netx_solution = np.array([row for row in solution])
        self.netx_solution = self.netx_solution[np.argsort(self.netx_solution[:,0])]
        return self.netx_solution
    
    def set_qubo_hyperparameters(self, t=8, rmin=64, rmax=127):
        """ Set the hyperparameters to use for the QUBO solver. """
        self.hyperparameters = {
            "temperature": int(t),
            "refract": np.random.randint(rmin, rmax, self.graph.number_of_nodes()),
            "refract_counter": np.random.randint(0, rmin, self.graph.number_of_nodes()),
        }

    def solve_with_lava_qubo(self, timeout=1000, probe_cost=False):
        """ Find a maximum independent set using QUBO in Lava. """
        self.lava_backend = 'Loihi2' if loihi.host else 'CPU'
        self.probe_cost = probe_cost
        qubo_matrix = MISProblem._get_qubo_cost_from_adjacency(
            self.adjacency, self.qubo_weights[0], self.qubo_weights[1])
        self.qubo_problem = QUBO(qubo_matrix)
        solver = OptimizationSolver(self.qubo_problem)
        self.solver_report = solver.solve(
            config=SolverConfig(
                timeout=timeout,
                hyperparameters=self.hyperparameters,
                target_cost=self.target_cost,
                backend=self.lava_backend,
                probe_cost=self.probe_cost,
            )
        )
        qubo_state = self.solver_report.best_state
        self.lava_solution = np.array(self.graph.nodes)[np.where(qubo_state)[0]]
        return self.lava_solution

    def plot_problem(self):
        """ Plot the problem state using pyplot. """
        plt.figure(figsize=(12,4), dpi=120)
        plt.subplot(131)
        plt.scatter(self.requests[:,0], self.requests[:,1], s=2)
        for y in self.view_coords:
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            verts = [[-0.05, y + self.view_height / 2],
                    [0.05, y + self.view_height],
                    [0.05, y + 0.0],
                    [-0.05, y + 0.0]]
            plt.gca().add_patch(PathPatch(Path(verts, codes), ec='none', alpha=0.3, fc='lightblue'))
            plt.scatter([-0.05], [y + self.view_height / 2], s=10, marker='s', c='gray')
            plt.plot([0, 1], [y + self.view_height / 2, y + self.view_height / 2], 'C1--', lw=0.75)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Schedule {self.num_satellites} satellites to observe {self.num_requests} targets.')
        plt.subplot(132)
        ntx.draw_networkx(self.graph, with_labels=False, node_size=2, width=0.5)
        plt.title(f'Infeasibility graph with {self.graph.number_of_nodes()} nodes.')
        plt.subplot(133)
        plt.imshow(self.adjacency, aspect='auto')
        plt.title(f'Adjacency matrix has {self.adjacency.mean():.2%} connectivity.')
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    def plot_solutions(self):
        """ Plot the solutions using pyplot. """
        plt.figure(figsize=(12,4), dpi=120)
        if self.netx_solution is not None:
            plt.subplot(131)
            plt.scatter(self.requests[:,0], self.requests[:,1], s=2, c='C1')
            for i in self.satellites:
                sat_plan = self.netx_solution[:,1] == i
                plt.plot(self.netx_solution[sat_plan,2], self.netx_solution[sat_plan,3], 'C0o-', markersize=2, lw=0.75)
            plt.title(f'NetworkX schedule satisfies {self.netx_solution.shape[0]} requests.')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(132)
        else:
            plt.subplot(121)
        plt.scatter(self.requests[:,0], self.requests[:,1], s=2, c='C1')
        for i in self.satellites:
            sat_plan = self.lava_solution[:,1] == i
            plt.plot(self.lava_solution[sat_plan,2], self.lava_solution[sat_plan,3], 'C0o-', markersize=2, lw=0.75)
        plt.title(f'Lava schedule satisfies {self.lava_solution.shape[0]} requests.')
        plt.xticks([])
        plt.yticks([])
        if self.solver_report.cost_timeseries is not None:
            plt.subplot(233)
            plt.plot(self.solver_report.cost_timeseries, lw=0.75)
            plt.title(f'QUBO solution cost is {self.solver_report.best_cost}')
            plt.subplot(236)
        else:
            plt.subplot(133)
        longest_plan = 1
        for i in self.satellites:
            sat_plan = self.lava_solution[:,1] == i
            longest_plan = max(longest_plan, sat_plan.sum() - 1)
            x = self.lava_solution[sat_plan,2]
            y = self.lava_solution[sat_plan,3]
            plt.plot(abs(np.diff(y) / np.diff(x)), lw=0.75)
        plt.plot([0, longest_plan], [self.turn_rate, self.turn_rate], '--', lw=0.75)
        plt.title(f'Satellite turn rates')
        plt.tight_layout()
        plt.show()
