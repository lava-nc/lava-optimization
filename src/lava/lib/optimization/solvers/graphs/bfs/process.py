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
    num_nodes_per_aggregator:

    num_nodes_per_distributor:

    """

    def __init__(
        self,
        connection_config: ConnectionConfig,
        num_nodes_per_aggregator=512,
        num_nodes_per_distributor=512,
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
            adjacency_matrix = adjacency_matrix,
            connection_config=connection_config,
            num_nodes_per_aggregator=num_nodes_per_aggregator,
            num_nodes_per_distributor=num_nodes_per_distributor,
            name=name,
            log_config=log_config,
        )

        # To initiate graph search provide (start_node_ID, target_node_ID)
        self.start_search = InPort(shape=(2,)) 
        
        # OutPort will be activated later for robotic implementation
        #self.shortest_path_nodes = OutPort(shape=(1,))
        
        # Only symmetric adjacency matrices are allowed here
        self._input_validation(adjacency_matrix)
        self.adjacency_matrix = Var(shape=adjacency_matrix.shape)
        self.shortest_path_value = Var(shape=1)
   


    def _input_validation(self, adjacency_matrix):
        assert (abs(adjacency_matrix-adjacency_matrix.T)>1e-10).nnz==0,f"Matrices" \
        f"need to be symmetric to continue with bfs"

class BFSNeuron(AbstractProcess):
    """The neurons that produce a single spiking wavefront to perform shortest
    path search. The start and destination neurons are set before this neuron 
    starts executing. Once this is done, the forward pass begins from the start 
    neuron through the adjacency matrix that the neurons are connected to. The 
    forwards passes continue till the destination neuron has been found and then
    the backward passes are initated through the same adjaceny matrix albeit 
    through different ports. During the backward pass, if the neuron is on the 
    shortest path, it sends a signal out through the s_out_spk_o ports to the 
    solution readout module. The IDs of the corresponding readout are thus 
    recieved on host which are in-turn processed to figure out the shortest path
    from start to destination in the correct sequence. 

    Parameters
    ----------

    shape : int tuple, optional
        A tuple defining the shape of the BFS neurons. Defaults to (1,). Is 
        usually equal to the number of graph nodes. 

    """
    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))

        # Ports
        # In/outPorts that come from/go to the adjacency matrix
        
        # Forward pass
        self.a_in_fwd = InPort(shape=(shape[0],))
        self.s_out_fwd = OutPort(shape=(shape[0],))
        
        # Backward pass
        self.a_in_bwd = InPort(shape=(shape[0],))
        self.s_out_bwd = OutPort(shape=(shape[0],))

        # Scaling ports
        self.a_in_dist = InPort(shape=(shape[0],))
        self.s_out_agg = OutPort(shape=(shape[0],))

        # I/O ports
        self.a_in_spk_i = InPort(shape=(shape[0],))
        self.s_out_spk_o = OutPort(shape=(shape[0],))

        # Constants used as flgs during execution
        self.dest_flg = 64     # 0b01000000
        self.start_flg = 128   # 0b10000000
        self.fwd_inc_done = 96 # 0b01100000

        # Vars for ProcModels
        self.counter_mem = Var(shape=shape, init=0)
        self.global_depth = Var(shape=shape, init=0)
        self.status_reg = Var(shape=shape, init=0)
