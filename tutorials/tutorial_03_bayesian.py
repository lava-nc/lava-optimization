# %% [markdown]
# *Copyright (C) 2021 Intel Corporation*<br>
# *SPDX-License-Identifier: BSD-3-Clause*<br>
# *See: https://spdx.org/licenses/*

# %% [markdown]
# ## Bayesian Optimization with Lava

# %% [markdown]
# This tutorial covers how to use the Bayesian Solver developed in Lava to optimize multi-dimensional black-box functions and demonstrates their use in larger process-based lava applications.

# %% [markdown]
# ### Recommended Tutorials before starting

# %% [markdown]
# - [Installing Lava](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial01_installing_lava.ipynb)
# - [Processes](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial02_processes.ipynb "Tutorial on Processes")
# - [ProcessModel](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial03_process_models.ipynb "Tutorial on ProcessModels")
# - [Execution](https://github.com/lava-nc/lava/blob/main/tutorials/in_depth/tutorial04_execution.ipynb "Tutorial on Executing Processes")

# %% [markdown]
# ### Why Bayesian Optimization?
# 
# Many state-of-the-art systems in the neuromorphic and greater scientific community have a plethora of hyperparameters that drastically effect the performance of a system. How to learn the trade-offs of changing different parameters has become a major research question with solutions comprising a variety of techniques such as evolutionary algorithms, quadratic programming, Bayesian optimization, etc.
# 
# **Bayes Theorem:**
# - $ P(B|A) = \frac{P(B|A)*P(A)}{P(B)} $
# 
# **Taking a closer look:**
# - $ A, B = \text{correlated events} $
# - $ P(A) = \text{independent probability of A} $
# - $ P(B) = \text{independent probability of B} $
# - $ P(A, B) = \text{the probability of } A \text{ given } B$
# - $ P(B, A) = \text{the probability of } B \text{ given } A$
# 
# Using the aforementioned theorem, Bayesian optimization creates an approximation of the underlying black-box function based on prior knowledge to predict the probability of subsequent events.
# 
# For information of the details of Bayesian optimization, *"A Tutorial on Bayesian Optimizer"* by Frazier, P.I. provides a great introduction.

# %% [markdown]
# ### A Bayesian Optimizer in Lava
# 
# To highlight the ability of the eventified Bayesian optimization system in Lava-Optimization, we will define a multi-dimensional, single-objective "black-box" function, create the appropriate processes and models, along with instantiating the solver.
# 
# For our test function, we'll use the Ackley function proposed by David Ackley in his dissertation as a test function for optimization problems:
# 
# $f(x_0, x_1) = -20\text{exp}[-0.2\sqrt{0.5 * (x_0^2+x_1^2)}]-\text{exp}[0.5(\text{cos}2\pi x_0 + \text{cos}2\pi x_1)] + e + 20$
# 
# *Ackley, D. H. (1987) "A connectionist machine for genetic hillclimbing", Kluwer Academic Publishers, Boston MA.*

# %% [markdown]
# ### Step 1) Importing all required packages

# %%

import math
import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.bayesian.solver import BayesianSolver


# %% [markdown]
# ### Step 2) Defining Lava process for the Ackley function

# %%
class AckleyFuncProcess(AbstractProcess):
    """Process defining the architecture of the Ackley function
    """
    def __init__(self, num_params: int = 2, num_objectives: int = 1,
        **kwargs) -> None:
        """initialize the AckleyFuncProcess

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

@implements(proc = AckleyFuncProcess, protocol = LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAckleyFuncProcessModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the Ackley function process.
    """

    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    num_params = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""
        x = self.x_in.recv()
        y = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x[0]**2 + x[1]**2)))
        y -= math.exp(0.5 * (math.cos(2 * math.pi * x[0]) + \
            math.cos(2 * math.pi * x[1])))
        y += math.e + 20

        output_length: int = self.num_params + self.num_objectives
        output = np.ndarray(shape=(output_length, 1))
        output[0][0], output[1][0], output[2][0] = x[0], x[1], y
        
        self.y_out.send(output)

def main():
    # %% [markdown]
    # ### Step 4) Creating Search Space

    # %% [markdown]
    # All hyperparameter searches are defined by their search spaces. The Bayesian Solver supports three types of parameter dimensions: integer, continuous, and categorical. All dimensions are specified with NumPy n-dimensional arrays of shape (5, \<number of dimensions\>). **It is extremely important that the number of dimensions of your parameter search space matches the input shape of your black-box process!**
    # 
    # For each of your parameters:
    # 1) specify the type of parameter, where the type shall be placed at index 0.
    #     - Integer = "integer"
    #     - Continuous = "continuous"
    #     - Categorical = "categorical"
    # 2) (integer/continuous-based parameters) define the minimum and maximum bounds
    #     - Minimum : np.float64 --> index 1
    #     - Maximum : np.float64 --> index 2
    #     - Categories : np.nan --> index 3
    # 3) (categorical parameters) we need to specify all of the categorical options 
    #     - Minimum : np.nan --> index 1
    #     - Maximum : np.nan --> index 2
    #     - Categories : list/ndarray[\<categories\>] --> index 3
    # 4) specify a unique identifier for the parameter
    #     - Name: str --> index 4

    # %%
    # given that the Ackley function accepts two continuous parameters, we will make
    # a search space that has one continuous dimension and one categorical dimension
    search_space: np.ndarray = np.array([
        ["continuous", np.float64(-5), np.float64(5), np.nan, "x0"],
        ["categorical", np.nan, np.nan, np.arange(-2, 5, 0.125), "x1"]
    ], dtype=object)


    # %% [markdown]
    # ### Step 5) Initialize Bayesian Solver
    # 
    # Given the complexity of the lower-level aspects of the Bayesian optimization process, the BayesianSolver has a complex configuration schema that needs to be followed.
    # 
    # #### Parameters:
    # 1) **acq_func_config** : dict
    #     - **"type":** str
    #         - *Summary:* specify the function to minimize over the posterior distribution
    #         - *Option 1:* "LCB" = lower confidence bound
    #         - *Option 2:* "EI" = negative expected improvement
    #         - *Option 3:* "PI" = negative probability of improvement
    #         - *Option 4:* "gp_hedge" = probabilistically determine which of the aforementioned functions to use at every iteration
    #         - *Option 5:* "EIps" = negative expected improved with consideration of the total function runtime
    #         - *Option 6:* "PIps" = negative probability of improvement while taking into account the total function runtime
    # 2) **acq_opt_config** : dict
    #     - **"type"** : str
    #         - *Summary:* specify the method to minimize the acquisition function
    #         - *Option 1:* "sampling" = random selection from the acquisition function
    #         - *Option 2:* "lbfgs" = inverse Hessian matrix estimation
    #         - *Option 3:* "auto" = automatically configure based on the search space
    # 3) **ip_gen_config** : dict
    #     - **"type"**: str
    #         - *Summary:* specify the method to explore the search space before the Gaussian regressor starts to converge
    #         - *Option 1:* "random" = uniform distribution of random numbers
    #         - *Option 2:* "sobol" = Sobol sequence
    #         - *Option 3:* "halton" = Halton sequence
    #         - *Option 4:* "hammersly" = Hammersly sequence
    #         - *Option 5:* "lhs" = latin hypercube sequence
    #         - *Option 6:* "grid" = uniform grid sequence
    # 4) **num_ips** : int
    #     - *Summary:* the number of points to explore with the initial point generator before using the regressor
    # 5) **seed** : int
    #     - *Summary:* An integer seed that sets the random state increases consistency in subsequent runs
    # 6) **est_config** : dict
    #     - **"type"**: str
    #         - *Summary:* specify the type of surrogate regressor to learn the search space:
    #         = *Option 1:* "GP" - gaussian process regressor
    # 7) **num_objectives** : int
    #     - *Summary:* specify the number of objectives to optimize over; currently limited to single objective

    # %%
    num_ips: int = 5
    seed: int = 0

    solver = BayesianSolver(
        acq_func_config={"type": "gp_hedge"},
        acq_opt_config={"type": "auto"},
        ip_gen_config={"type": "random"},
        num_ips=num_ips,
        seed=seed
    )

    # %% [markdown]
    # ### Step 6) Solve the black-box process

    # %%
    # Now we are at the final stages of the process! Before we can solve the problem, we need to do a few things:

    # 1) initialize the Ackley function process
    problem = AckleyFuncProcess()

    # 2) specify the experiment name and the number of optimization iteration
    experiment_name: str = "bayesian_tutorial_results"
    num_iter: int = 10

    # 3) solve the problem!
    solver.solve(
        name=experiment_name,
        num_iter=num_iter,
        problem=problem,
        search_space=search_space,
    )

    # %% [markdown]
    # ### Moving Forward
    # 
    # We appreciate you taking the time to follow this tutorial! :)
    #
    # Now you have seen how to create and solve a black-box function using the eventified Bayesian solver in Lava!
    # 
    # If you are interested in learning more about the details of the specific implementation of the solver and
    # the lower-level processes, have a look at the [Lava documentation](https://lava-nc.org/ "Lava Documentation") or dive into the [source code](https://github.com/lava-nc/lava-optimization/tree/main/src/lava/lib/optimization/solvers/qp
    # "QP source code").
    # 
    # To receive regular updates on the latest developments and releases of the Lava Software Framework please subscribe to the [INRC newsletter](http://eepurl.com/hJCyhb "INRC Newsletter").

if __name__ == "__main__":
    main()
