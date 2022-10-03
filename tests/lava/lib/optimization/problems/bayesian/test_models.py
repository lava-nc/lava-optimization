# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.problems.bayesian.models import (
    DualInputFunction,
    SingleInputFunction
)


class InputParamVecProcess(AbstractProcess):
    def __init__(self, num_params: int, spike: np.ndarray, **kwargs) -> None:
        """Process to set an input parameter vector to evaluate black-box
        function accuracy

        num_params : int
            the number of parameters to send to the test function
        spike : np.ndarray
            the parameter vector to send to the black-box process
        """
        super().__init__(**kwargs)

        self.x_out = OutPort(shape=(num_params, 1))
        self.data = Var(shape=(num_params, 1), init=spike)


class OutputPerfVecProcess(AbstractProcess):
    def __init__(self, num_params: int, num_objectives: int,
                 **kwargs) -> None:
        """Process to validate the resulting performance vector from the
        black-box function

        num_params : int
            the number of parameters within each performance vector
        num_objectives : int
            the number of objectives within each performance vector
        valid_spike : np.ndarray
            the expected performance vector
        """
        super().__init__(**kwargs)

        perf_vec_length: int = num_params + num_objectives
        self.y_in = InPort(shape=(perf_vec_length, 1))
        self.recv_data = Var(shape=(perf_vec_length, 1))


@implements(proc=InputParamVecProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyInputParamVecModel(PyLoihiProcessModel):
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    data: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self) -> None:
        """send the test data to the black-box process"""
        self.x_out.send(self.data)


@implements(proc=OutputPerfVecProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputPerfVecProcess(PyLoihiProcessModel):
    y_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    recv_data: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self) -> None:
        """receive the result vector from the black-box process"""
        self.recv_data = self.y_in.recv()


class TestModels(unittest.TestCase):
    """Tests all model behaviors associated with the Bayesian problems

    Refer to Bayesian models.py to learn more about behaviors.
    """

    def test_model_dual_cont_input_func(self) -> None:
        """test behavior of the DualInputFunction process"""

        input_spike = np.ndarray((2, 1), buffer=np.array([0.1, 0.1]))
        valid_spike = np.array([0.1, 0.1, 1.00540399861])

        input_probe = InputParamVecProcess(num_params=2, spike=input_spike)
        bb_process = DualInputFunction()
        output_probe = OutputPerfVecProcess(num_params=2, num_objectives=1)

        input_probe.x_out.connect(bb_process.x_in)
        bb_process.y_out.connect(output_probe.y_in)

        output_probe.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg()
        )

        result: np.ndarray = output_probe.recv_data.get()
        self.assertEqual(result[0][0], valid_spike[0])
        self.assertEqual(result[1][0], valid_spike[1])
        self.assertAlmostEqual(result[2][0], valid_spike[2])

        output_probe.stop()

    def test_model_single_input_nonlinear_func(self) -> None:
        """test behavior of the SingleInputFunction process"""

        input_spike = np.array([5])
        valid_spike = np.array([5, 0.727989444555])

        input_probe = InputParamVecProcess(num_params=1, spike=input_spike)
        bb_process = SingleInputFunction()
        output_probe = OutputPerfVecProcess(num_params=1, num_objectives=1)

        input_probe.x_out.connect(bb_process.x_in)
        bb_process.y_out.connect(output_probe.y_in)

        output_probe.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg()
        )

        result: np.ndarray = output_probe.recv_data.get()
        self.assertEqual(result[0][0], valid_spike[0])
        self.assertAlmostEqual(result[1][0], valid_spike[1])

        output_probe.stop()


if __name__ == "__main__":
    unittest.main()
