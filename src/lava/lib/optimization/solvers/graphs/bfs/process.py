# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import typing as ty
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.connection_config import ConnectionConfig


class BreadthFirstSearch(AbstractProcess):
    r"""Process which implements bread first search on unweighted & undirected 
    graphs in order to find the shortest path from start node to target node. 
    This process encapsulates the solver and Spike I/O (to set target and 
    destination nodes and to readout solutions).

    Attributes
    ----------
    start_search: InPort
        Recieves starting node and target node from the superhost.
    shortest_path_nodes: OutPort
        Shape of the OutPort is (1, ). Readout only one the spike counters when 
        shortest path/s is/are found. In the event of multiple shortest paths, 
        connectivity with the previously read out solution nodes are checked for
        before reading out the state of the spike counter. 
    adjacency_matrix: np.ndarray
        Defines the connectivity of the graph (unweighted/undirected).
    """

    def __init__(
        self,
        shape: ty.Tuple[int, ...],
        connection_config: ConnectionConfig,
        adjacency_matrix = None,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        shape: tuple
            A tuple of the form (number of variables, domain size).

        name: str, optional
            Name of the Process. Default is 'Process_ID', where ID is an integer
            value that is determined automatically.
        log_config: LogConfig, optional
            Configuration options for logging.z"""

        super().__init__(
            shape=(1,),
            adjacency_matrix = adjacency_matrix,
            connection_config=connection_config,
            name=name,
            log_config=log_config,
        )

        # To initiate graph search provide (start_node_ID, target_node_ID)
        self.start_search = InPort(shape=(2,)) 
        self.shortest_path_nodes = OutPort(shape=(1,))
        # Only symmetric adjacency matrices are allowed here
        self._input_validation(adjacency_matrix)
        self.adjacency_matrix = Var(shape=adjacency_matrix.shape)
   


    def _input_validation(self, adjacency_matrix):
     assert (abs(adjacency_matrix-adjacency_matrix.T)>1e-10).nnz==0,f"Matrices"\ 
     f"need to be symmetric to continue with bfs"

