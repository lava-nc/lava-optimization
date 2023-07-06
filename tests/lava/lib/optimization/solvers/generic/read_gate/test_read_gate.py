# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from tests.lava.test_utils.utils import Utils

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.receiver.models import ReceiverModel
from lava.proc.spiker.process import Spiker
import typing as ty

from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.process.process import AbstractProcess, LogConfig

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

from lava.proc.receiver.process import Receiver

try:
    from lava.lib.optimization.solvers.generic.monitoring_processes\
        .solution_readout.process import SolutionReadout
    from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
    from lava.proc.spiker.models import SpikerModel
except ImportError:
    class SolutionReadout:
        pass

    class ReadGate:
        pass

    class ReadGateCModel:
        pass

    class SpikerNcModel:
        pass

run_loihi_tests: bool = Utils.get_bool_env_setting("RUN_LOIHI_TESTS")


class Pauser(AbstractProcess):
    """Process that upon receving an input signal requests execution to pause.

    Parameters
    ----------
    name: Name of the Process. Default is 'Process_ID', where ID is an integer
        value that is determined automatically.
    log_config: Configuration options for logging.

    InPorts
    -------
    req_stop_rcv: Receives a signal from downstream process to request execution
        to pause.
    """

    def __init__(self,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(name=name,
                         log_config=log_config)
        self.timestep_in = InPort(shape=(1,))


@implements(proc=Pauser, protocol=AsyncProtocol)
@requires(CPU)
class PauserModel(PyLoihiProcessModel):
    """CPU model for the Pause process.

    Request execution to pause Upon receiving a non-zero input on the
    req_stop_rcv input port.
    """
    timestep_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int32, precision=32
    )
    stop = False

    def run_spk(self):
        """Stop if requested."""
        if self.stop:
            return
        timestep = self._receive_data()
        self._stop_if_requested(timestep)

    def _receive_data(self):
        timestep = self.timestep_in.recv()[0]
        return timestep

    def _stop_if_requested(self, timestep):
        if timestep > 0 or timestep == -1:
            self.stop = True


@unittest.skipUnless(run_loihi_tests, "")
class TestReadGate(unittest.TestCase):
    """Test ReadGate process.

    A Spiker process of shape (1,) is used to send messages with a payload as
    cost to the ReadGate process, another Spiker process with shape (4,)
    is used as a proxy of the main population holding the state that
    produced the cost. The cost_receiver and solution_receiver process are
    upstream processes to which the ReadGate forwards its outputs. It is
    expected that ReadGate:
     - Detects when a new cost was received.
     - Forwards the new cost to its output port because is better than before.
     - Reads and forwards the network state when improved cost is detected.
    """
    run_lib_tests: bool = not (Utils.get_bool_env_setting("RUN_LIB_TESTS"))
    skip_message_loihi = 'Library tests not set to run'

    def run_test(self,
                 payload_cost_last_bytes,
                 payload_cost_first_byte) -> ty.Tuple[np.ndarray, np.ndarray]:
        from lava.lib.optimization.solvers.generic.read_gate.process import (
            ReadGate,
        )
        # Create processes.
        target_cost = -5
        integrator_first = Spiker(shape=(1,), period=7,
                                  payload=payload_cost_first_byte)
        integrator_last = Spiker(shape=(1,), period=7,
                                 payload=payload_cost_last_bytes)
        readgate = ReadGate(shape=(1,), target_cost=target_cost)
        solution_readout = SolutionReadout(shape=(1,), target_cost=target_cost)

        # Connect ports.
        integrator_first.s_out.connect(readgate.cost_in_first_byte_0)
        integrator_last.s_out.connect(readgate.cost_in_last_bytes_0)
        readgate.cost_out.connect(solution_readout.cost_in)
        readgate.solution_out.connect(solution_readout.read_solution)
        readgate.send_pause_request.connect(solution_readout.timestep_in)

        # Configure execution.
        from lava.lib.optimization.solvers.generic.read_gate.models import (
            get_read_gate_model_class,
        )

        ReadGatePyModel = get_read_gate_model_class(num_in_ports=1)
        pdict = {ReadGate: ReadGatePyModel, Spiker: SpikerModel,
                 Receiver: ReceiverModel}
        run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)

        readgate._log_config.level = 20

        # Run and gather data.
        received_state = []
        received_cost = []
        spiker_counter = []
        # for num_steps in range(1, 16):
        num_steps = 10

        readgate.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_cfg)
        state = solution_readout.solution.get()[0]
        cost = solution_readout.min_cost.get()[0]
        counter = integrator_last.counter.get()[0]
        received_state.append(state)
        received_cost.append(cost)
        spiker_counter.append(counter)
        readgate.stop()

        # Verify Integrator's dynamics is correct.
        expected_counter = [3]
        self.assertTrue(np.all(spiker_counter == expected_counter))
        return received_cost, received_state

    @unittest.skipIf(run_lib_tests, skip_message_loihi)
    def test_sends_message_if_cost_changes(self):
        payload_cost_last_bytes = 0
        payload_cost_first_byte = -1
        cost, state = self.run_test(
            payload_cost_last_bytes=payload_cost_last_bytes,
            payload_cost_first_byte=payload_cost_first_byte)
        expected = [(payload_cost_first_byte << 24) + payload_cost_last_bytes]

        self.assertTrue(cost == expected)

    @unittest.skipIf(run_lib_tests, skip_message_loihi)
    def test_sends_solution_if_cost_changes(self):
        payload_cost_last_bytes = 16777206
        payload_cost_first_byte = -1
        cost, state = self.run_test(
            payload_cost_last_bytes=payload_cost_last_bytes,
            payload_cost_first_byte=payload_cost_first_byte)
        expected = [1]
        self.assertTrue(state == expected)

    @unittest.skipIf(run_lib_tests, skip_message_loihi)
    def test_sends_message_if_handles_32bit(self):
        """Test if ReadGate correctly handles 32bit cost. 32bit workloads are
        communicated by CostIntegrator using two spikes, the first communicating
        the first 8bit, the other the remaining 24bit. """

        payload_cost_last_bytes_arr = [0, 16777215, 16777215, 0, 16777206, 0,
                                       2**24 - 1, 0, 2**24 - 1, 0, 2**24 - 1,
                                       2**24 - 2]
        payload_cost_first_byte_arr = [-128, -127, -2, -2, -1, -1, 0, 1, 1,
                                       2, 2, 127]
        for ii in range(len(payload_cost_last_bytes_arr)):
            payload_cost_first_byte = payload_cost_first_byte_arr[ii]
            payload_cost_last_bytes = payload_cost_last_bytes_arr[ii]
            cost, state = self.run_test(
                payload_cost_last_bytes=payload_cost_last_bytes,
                payload_cost_first_byte=payload_cost_first_byte)
            expected = [(payload_cost_first_byte << 24)
                        + payload_cost_last_bytes]
            self.assertTrue(cost == expected)
