# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import math
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.problems.bayesian.processes import (
    DualInputFunction,
    SingleInputFunction
)


@implements(proc=SingleInputFunction, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySingleInputFunctionModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the SingleInput process that represents a
    single input/output non-linear objective function.
    """

    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    num_params = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""
        x = self.x_in.recv()
        y = math.cos(x) * math.sin(x) + (x * x / 25)

        output_length: int = self.num_params + self.num_objectives
        output = np.ndarray(
            shape=(output_length, 1),
            buffer=np.array([x, y])
        )

        self.y_out.send(output)


@implements(proc=DualInputFunction, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyDualInputFunctionModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the DualInputFunction process that
    represents a dual continuous input, single output, non-linear objective
    function.
    """

    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    num_params = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""

        x = self.x_in.recv()
        y = math.sin(x[1] * x[0]) + (0.2 * x[0]) ** 2 + math.cos(x[1])

        output_length: int = self.num_objectives + self.num_params
        output = np.ndarray(
            shape=(output_length, 1),
            buffer=np.array([x[0], x[1], y])
        )

        self.y_out.send(output)
