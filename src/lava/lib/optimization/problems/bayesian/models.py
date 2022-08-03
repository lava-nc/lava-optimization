import math
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.problems.bayesian.processes import (
    DualContInputFunction,
    SingleInputLinearFunction,
    SingleInputNonLinearFunction
)

@implements(proc = SingleInputLinearFunction, protocol = LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyLinearTestFunctionModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the SingleInputLinearFunction process
    that represents a single input/output linear function.
    """

    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    num_params = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""
        x = self.x_in.recv()
        y: float = x * 8 + 9

        output_length: int = self.num_objectives + self.num_params
        output = np.ndarray(
            shape=(output_length, 1),
            buffer=np.array([x, y])
        )

        self.y_out.send(output)


@implements(proc = SingleInputNonLinearFunction, protocol = LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyNonLinearTestFunctionModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the SingleInputNonLinear process
    that represents a single input/output non-linear objective function.
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


@implements(proc = DualContInputFunction, protocol = LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyDualContInputFunctionModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the DualContInputFunction process
    that represents a dual continuous input, single output, non-linear
    objective function.
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
