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
    node_description:  np.array
        A numpy array that specifies which of the neurons is a start neuron or
        a destination neuron. The start neuron position has a code of 128 and
        the destination neuron has a code of 64. There cannot be more than 1
        start and destination neurons each. All other neurons should be set to
        status 0. Ensure that there is atleast one destination node and start
        node.
    adjacency_matrix: np.ndarray
        Defines the connectivity of the graph (unweighted/undirected).
    num_nodes_per_aggregator: int
        The number of graph nodes that can connect to one aggregator neuron.
        The limit to this is based on fan-in constraints to a core on Loihi

    num_nodes_per_distributor: int
        The number of graph nodes that can connect to one aggregator neuron.
        The limit to this is based on fan-out constraints from a core on Loihi
    """

    def __init__(
        self,
        connection_config: ConnectionConfig,
        node_description=None,
        num_nodes_per_aggregator=512,
        num_nodes_per_distributor=512,
        adjacency_matrix=None,
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
            adjacency_matrix=adjacency_matrix,
            node_description=node_description,
            connection_config=connection_config,
            num_nodes_per_aggregator=num_nodes_per_aggregator,
            num_nodes_per_distributor=num_nodes_per_distributor,
            name=name,
            log_config=log_config,
        )

        # To initiate graph search provide (start_node_ID, target_node_ID)
        self.start_search = InPort(shape=(2,))

        # OutPort will be activated later for robotic implementation
        # self.shortest_path_nodes = OutPort(shape=(1,))

        # Specify target and destination neurons
        self.node_description = node_description

        # Only symmetric adjacency matrices are allowed here
        valid_input = self._input_validation(adjacency_matrix)

        if valid_input:
            self.adjacency_matrix = Var(
                shape=adjacency_matrix.shape, init=adjacency_matrix
            )
        else:
            raise ValueError(
                "Matrix is either not symmetric or contains values\
                             that are not 0 or 1"
            )

    def _input_validation(self, matrix):
        """
         Check if all elements are either 0 or 1. Also if the matrix is \
        symmetric
         """
        return np.array_equal(matrix, matrix.T) and not (
            np.any(matrix > 1) or np.any(matrix > 1)
        )


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
        A tuple defining the shape of the BFS neurons. Defaults to None. Is
        usually equal to the number of graph nodes.
    status: np.array
        A numpy array that specifies which of the neurons is a start neuron or
        a destination neuron. The start neuron position has a code of 128 and
        the destination neuron has a code of 64. There cannot be more than 1
        start and destination neurons each. All other neurons should be set to
        status 0. Ensure that there is atleast one destination node and start
        node.

    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)

        shape = kwargs.pop("shape", (None,))

        # start node code = 128
        # dest node code = 64
        status = kwargs.pop("status", 0)
        self._check_status_sanity(status)

        # Ports
        # In/outPorts that come from/go to the adjacency matrix

        # Forward pass
        self.a_in_1 = InPort(shape=(shape[0],))
        self.s_out_1 = OutPort(shape=(shape[0],))

        # Backward pass
        self.a_in_2 = InPort(shape=(shape[0],))
        self.s_out_2 = OutPort(shape=(shape[0],))

        # Scaling ports
        self.a_in_3 = InPort(shape=(shape[0],))
        self.s_out_3 = OutPort(shape=(shape[0],))

        # I/O ports
        self.a_in_4 = InPort(shape=(shape[0],))
        self.s_out_4 = OutPort(shape=(shape[0],))

        # Constants used as flgs during execution
        self.dest_flag = Var(shape=(1,), init=64)  # 0b01000000
        self.start_flag = Var(shape=(1,), init=128)  # 0b10000000
        self.fwd_inc_done = Var(shape=(1,), init=32)  # 0b00100000

        # Vars for ProcModels
        self.counter_mem = Var(shape=shape, init=0)
        self.global_depth = Var(shape=shape, init=0)
        self.status_reg = Var(shape=shape, init=status)

    def _check_status_sanity(self, status):
        """
        helper function to ensure that there are not more than 1 neuron marked as
        start or destination. Also to ensure that all the other neurons are 0.
        Function also ensures that there is atleast one destination node and
        start node
        """
        allowed_values = [0, 64, 128]
        contains_other_values = np.any(~np.isin(status, allowed_values))
        num_destinations = len(np.where(status == 64)[0])
        num_starts = len(np.where(status == 128)[0])
        assert (
            num_destinations == 1
        ), f"You have either no destination node or more than 1 destination node"
        assert (
            num_starts == 1
        ), f"You have either no start node or more than 1 start node"
        assert (
            contains_other_values == False
        ), f"Status contains values that are not permitted. Please check that your array contains only 0, 64, and 128"
