# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import unittest

from lava.lib.optimization.problems.bayesian.processes import (
    BaseObjectiveFunction,
    SingleInputFunction,
    DualInputFunction,
)


class TestProcesses(unittest.TestCase):
    """Initialization Tests of all processes of the Bayesian problems

    All tests check if the vars are properly assigned in the process and if
    the ports are shaped properly. Refer to Bayesian models.py to understand
    behaviors.
    """

    def test_process_base_objective_func(self) -> None:
        """test initialization of the BaseObjectiveFunction process"""

        num_params: int = 5
        num_objectives: int = 2

        process = BaseObjectiveFunction(
            num_params=num_params,
            num_objectives=num_objectives
        )

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of internal variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of internal variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )

    def test_process_single_input_nonlinear_func(self) -> None:
        """test initialization of the SingleInputNonLinearFunction process"""

        num_params: int = 1
        num_objectives: int = 1

        process = SingleInputFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of internal variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of internal variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )

    def test_process_dual_cont_input_func(self) -> None:
        """test initialization of the DualContInputFunction process"""

        num_params: int = 2
        num_objectives: int = 1

        process = DualInputFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of internal variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of internal variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )


if __name__ == "__main__":
    unittest.main()
