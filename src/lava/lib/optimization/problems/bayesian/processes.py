# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class BaseObjectiveFunction(AbstractProcess):
    """
    A base objective function process that shall be used as the basis of
    all black-box processes.
    """

    def __init__(self, num_params: int, num_objectives: int,
                 **kwargs) -> None:
        """initialize the BaseObjectiveFunction

        Parameters
        ----------
        num_params : int
            an integer specifying the number of parameters within the
            search space
        num_objectives : int
            an integer specifying the number of qualitative attributes
            used to measure the black-box function
        """
        super().__init__(**kwargs)

        # Internal State Variables
        self.num_params = Var((1,), init=num_params)
        self.num_objectives = Var((1,), init=num_objectives)

        # Input/Output Ports
        self.x_in = InPort((num_params, 1))
        self.y_out = OutPort(((num_params + num_objectives), 1))


class SingleInputFunction(BaseObjectiveFunction):
    """
    An abstract process representing a single input/output test function.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the process with the associated parameters"""
        super().__init__(num_params=1, num_objectives=1, **kwargs)


class DualInputFunction(BaseObjectiveFunction):
    """
    An abstract process representing a dual input, single output
    test function.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the process with the associated parameters"""
        super().__init__(num_params=2, num_objectives=1, **kwargs)
