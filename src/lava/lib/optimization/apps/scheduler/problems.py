# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


from typing import Optional, Union
import numpy as np
import networkx as ntx

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path


class SchedulingProblem:
    def __init__(self,
                 num_agents: int = 3,
                 num_tasks: int = 3,
                 sat_cutoff: Union[float, int] = 0.99):
        """Schedule `num_tasks` tasks among `num_agents` agents such that
        every agent performs exactly one task and every task gets assigned
        exactly one agent.

        Parameters
        ----------
        num_agents (int) : number of agents available to perform all tasks.
        Default is arbitrarily chosen as 3.

        num_tasks (int) : number of tasks to be performed. Default is
        arbitrarily chosen as 3.

        sat_cutoff (float or int) : If float, this is the satisfiability
        cut-off calculated as the ratio between the number of tasks that get
        assigned an agent to the total number of tasks. A fraction between 0
        and 1. If int, this is the target cost for the underlying QUBO
        solver. Default is 0.99 (i.e., 99% of the total number of tasks get
        assigned an agent).
        """
        self._num_agents = num_agents
        self._agent_ids = range(num_agents)
        self._agent_attrs = None
        self._num_tasks = num_tasks
        self._task_ids = range(num_tasks)
        self._task_attrs = None
        self._sat_cutoff = sat_cutoff
        self.graph = None
        self.adjacency = None
        self.random_seed = None

    @property
    def num_agents(self):
        return self._num_agents

    @num_agents.setter
    def num_agents(self, val: int):
        self._num_agents = val

    @property
    def agent_ids(self):
        return self._agent_ids

    @property
    def agent_attrs(self):
        return self._agent_attrs

    @agent_attrs.setter
    def agent_attrs(self, attr_vec):
        self._agent_attrs = attr_vec

    @property
    def num_tasks(self):
        return self._num_tasks

    @num_tasks.setter
    def num_tasks(self, val: int):
        self._num_tasks = val

    @property
    def task_ids(self):
        return self._task_ids

    @property
    def task_attrs(self):
        return self._task_attrs

    @task_attrs.setter
    def task_attrs(self, attr_vec):
        self._task_attrs = attr_vec

    @property
    def sat_cutoff(self):
        return self._sat_cutoff

    @sat_cutoff.setter
    def sat_cutoff(self, val: float):
        self._sat_cutoff = val

    def is_node_valid(self, *args):
        return True

    def is_edge_conflicting(self, node1, node2):
        nodes = self.graph.nodes
        is_same_agent = (nodes[node1]["agent_id"] == nodes[node2]["agent_id"])
        is_same_task = (nodes[node1]["task_id"] == nodes[node2]["task_id"])
        return True if is_same_agent or is_same_task else False

    def generate(self, seed=None):
        """ Generate a new scheduler problem. """
        if seed:
            self.random_seed = seed
            np.random.seed(seed)
        self.graph = ntx.Graph()
        self.generate_valid_nodes()
        self.generate_edges_from_constraints()
        self.rescale_adjacency()

    def generate_valid_nodes(self):
        node_id = 0
        agent_id_attr_map = dict(zip(self.agent_ids, self.agent_attrs))
        task_id_attr_map = dict(zip(self.task_ids, self.task_attrs))
        for aid, a_attr in agent_id_attr_map.items():
            for tid, t_attr in task_id_attr_map.items():
                if self.is_node_valid(aid, tid):
                    self.graph.add_node(node_id,
                                        agent_id=aid,
                                        task_id=tid,
                                        agent_attr=a_attr,
                                        task_attr=t_attr)
                    node_id += 1

    def generate_edges_from_constraints(self):
        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                not_same = n1 != n2
                is_conflict = self.is_edge_conflicting(n1, n2)
                if not_same and is_conflict:
                    self.graph.add_edge(n1, n2)
        self.adjacency = ntx.to_numpy_array(self.graph)

    def rescale_adjacency(self):
        """ Scale the adjacency matrix weights for QUBO solver. """
        self.adjacency = np.triu(self.adjacency)
        self.adjacency += self.adjacency.T - 2 * np.diag(
            self.adjacency.diagonal())


class SatelliteScheduleProblem(SchedulingProblem):
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
            num_satellites: int = 6,
            view_height: float = 0.25,
            view_coords: Optional[np.ndarray] = None,
            num_requests: int = 48,
            turn_rate: float = 2,
            solution_criteria: float = 0.99,
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
        solution_criteria : float, default = 0.99
            The target for a successful solution. The solver will stop looking for
            a better schedule if the specified fraction of requests are satisfied.
        """
        super(SatelliteScheduleProblem,
              self).__init__(num_agents=num_satellites,
                             num_tasks=num_requests,
                             sat_cutoff=solution_criteria)
        self.num_satellites = self.num_agents
        self.num_requests = self.num_tasks

        self.view_height = view_height
        if view_coords is None:
            self.view_coords = np.linspace(0 - view_height / 2,
                                           1 - view_height / 2,
                                           num_satellites)
        else:
            self.view_coords = view_coords
        self.agent_attrs = list(zip([view_height] * num_satellites,
                                    self.view_coords))
        self.satellites = self.agent_ids
        self.turn_rate = turn_rate
        self.requests = None
        self.qubo_problem = None
        self.generate_requests()

    def generate_requests(self):
        """ Generate a random set of requests in the 2D plane. """
        self.requests = np.random.random((self.num_requests, 2))
        order = np.argsort(self.requests[:, 0])
        self.requests = self.requests[order, :]
        self.task_attrs = self.requests.tolist()

    def is_node_valid(self, sat_id, req_id):
        """ Return whether the request is visible to the satellite. """
        view_height = self.agent_attrs[sat_id][0]
        satellite_y_coord = self.agent_attrs[sat_id][1]
        request_y_coord = self.task_attrs[req_id][1]
        return (satellite_y_coord <= request_y_coord <= satellite_y_coord +
                view_height)

    def is_req_reachable(self, n1, n2):
        nodes = self.graph.nodes
        n1_req_coords = nodes[n1]["task_attr"]
        n2_req_coords = nodes[n2]["task_attr"]
        delta_x = abs(n1_req_coords[0] - n2_req_coords[0])
        delta_y = abs(n1_req_coords[1] - n2_req_coords[1])
        return self.turn_rate * delta_x >= delta_y

    def is_edge_conflicting(self, node1, node2):
        nodes = self.graph.nodes
        is_same_satellite = (nodes[node1]["agent_id"] == nodes[node2][
            "agent_id"])
        is_same_request = (nodes[node1]["task_id"] == nodes[node2]["task_id"])
        return is_same_request or (is_same_satellite and not
                                   self.is_req_reachable(node1, node2))

    def plot_problem(self):
        """ Plot the problem state using pyplot. """
        plt.figure(figsize=(12, 4), dpi=120)
        plt.subplot(131)
        plt.scatter(self.requests[:, 0],
                    self.requests[:, 1],
                    s=2)
        for y in self.view_coords:
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            verts = [[-0.05, y + self.view_height / 2],
                     [0.05, y + self.view_height],
                     [0.05, y + 0.0],
                     [-0.05, y + 0.0]]
            plt.gca().add_patch(
                PathPatch(Path(verts, codes), ec='none', alpha=0.3,
                          fc='lightblue'))
            plt.scatter([-0.05], [y + self.view_height / 2],
                        s=10, marker='s', c='gray')
            plt.plot([0, 1],
                     [y + self.view_height / 2,
                      y + self.view_height / 2],
                     'C1--', lw=0.75)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f'Schedule {self.num_satellites} satellites to observe '
            f'{self.num_requests} targets.')
        plt.subplot(132)
        ntx.draw_networkx(self.graph, with_labels=False,
                          node_size=2, width=0.5)
        plt.title(
            f'Infeasibility graph with {self.graph.number_of_nodes()} nodes.')
        plt.subplot(133)
        plt.imshow(self.adjacency, aspect='auto')
        plt.title(
            f'Adjacency matrix has {self.adjacency.mean():.2%} '
            f'connectivity.')
        plt.yticks([])
        plt.tight_layout()
        plt.show()
