from itertools import product
import numpy as np
import random
import sys
import unittest

from lava.lib.optimization.problems.bayesian.processes import (
    BaseObjectiveFunction,
    SingleInputLinearFunction,
    SingleInputNonLinearFunction,
    DualContInputFunction,
)

class TestProcesses(unittest.TestCase):
    """Initialization Tests of all processes of the Bayesian solver
    
    All tests check if the vars are properly assigned in the process abd if
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

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )

    def test_process_single_input_linear_func(self) -> None:
        """test initialization of the SingleInputLinearFunction process"""

        num_params: int = 1
        num_objectives: int = 1
        
        process = SingleInputLinearFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
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
        
        process = SingleInputNonLinearFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
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
        
        process = DualContInputFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
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