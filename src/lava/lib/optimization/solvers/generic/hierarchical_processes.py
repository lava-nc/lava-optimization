# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from numpy import typing as npt


class ContinuousVariablesProcess(AbstractProcess):
    """Process which implementation holds the evolution of continuous
    variables on the solver of an optimization problem."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class DiscreteVariablesProcess(AbstractProcess):
    r"""Process which implementation holds the evolution of discrete variables
    on the solver of an optimization problem.

    Parameters
    ----------
    shape: a tuple of the form (number of variables, domain size).
    step_size: The coefficient of the variables indicating their relative
        linear contribution to the cost function. Usually corresponds to the
        linear term or the diagonal of the quadratic term on the cost function.
    cost_diagonal: The diagonal of the coefficient of the quadratic term on the
        cost function.
    name: Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config: Configuration options for logging.z

    InPorts
    -------
    a_in: The addition of all inputs (per dynamical system) at this timestep
        will be received by this port.

    OutPorts
    --------
    s_out: The payload to be exchanged between the underlying dynamical systems
        when these fire.
    local_cost: the cost components per dynamical system underlying these
        variables, i.e., c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through
        this port. The cost integrator will then complete the cost computation
         by adding all contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.

    Vars
    ----
    variable_assignment: Holds the current value assigned to the variables by
        the solver network.
    """

    def __init__(self, shape: ty.Tuple[int, ...],
                 cost_diagonal: npt.ArrayLike = None,
                 hyperparameters: ty.Dict[str, ty.Union[int,
                                                        npt.ArrayLike]] = None,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         cost_diagonal=cost_diagonal,
                         name=name,
                         log_config=log_config)
        self.num_variables = np.prod(shape)
        self.hyperparameters = hyperparameters
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.variable_assignment = Var(shape=shape)
        self.cost_diagonal = cost_diagonal
        self.local_cost = OutPort(shape=shape)


class CostConvergenceChecker(AbstractProcess):
    """Process that continuously monitors cost convergence.

    Parameters
    ----------
    shape: The expected shape of the input cost components.
    name: Name of the Process. Default is 'Process_ID', where ID is an
    integer value that is determined automatically.
    log_config: Configuration options for logging.


    InPorts
    -------
    cost_components: Additive contributions to the total cost.

    OutPorts
    --------
    update_buffer: OutPort which notifies the next process about the
    detection of a better cost.

    Vars
    ----
    min_cost: Current minimum cost, i.e., the lowest reported cost so far.

    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None, ) -> None:
        super().__init__(shape=shape,
                         name=name,
                         log_config=log_config)
        self.shape = shape
        self.min_cost = Var(shape=(1,))
        self.cost_components = InPort(shape=shape)
        self.update_buffer = OutPort(shape=(1,))


class SatConvergenceChecker(AbstractProcess):
    """Process that continuously monitors satisfiability convergence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.satisfaction = None
        raise NotImplementedError


class AugmentedTermsProcess(AbstractProcess):
    """Process implementing cost coefficients as synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class ContinuousConstraintsProcess(AbstractProcess):
    """Process implementing continuous constraints via neurons and synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class DiscreteConstraintsProcess(AbstractProcess):
    """Process implementing discrete constraints via synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class MixedConstraintsProcess(AbstractProcess):
    """Process implementing continuous constraints via neurons and synapses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError


class StochasticIntegrateAndFire(AbstractProcess):
    r"""Event-driven stochastic discrete dynamical system with two outputs.

    The main output is intended as input to other dynamical systems on
    the network, whilst the second output is to transfer local information to be
    integrated by an auxiliary dynamical system or circuit.

    Parameters
    ----------
    shape: The shape of the set of dynamical systems to be created.
    init_state: The starting value of the state variable.
    step_size: a value to be added to the state variable at each timestep.
    noise_amplitude: The width/range for the stochastic perturbation to the
    state variable. A random number within this range will be added to the
    state variable at each timestep.
    input_duration: Number of timesteps by which each input should be preserved.
    min_state: The minimum value for the state variable. The state variable
    will be truncated at this value if updating results in a lower value.
    min_integration: The minimum value for the total input (addition of all
    valid inputs at a given timestep). The total input value will be truncated
    at this value if adding current and preserved inputs results in a lower
    value.
    steps_to_fire: After how many timesteps would the dynamical system fire
    and reset without stochastic perturbation. Note that if noise_amplitude > 0,
    the system will stochastically deviate from this value.
    refractory_period: number of timesteps to wait after firing and reset before
    resuming updating.
    cost_diagonal: The linear coefficients on the cost function of the
    optimization problem where this system will be used.
    name: Name of the Process. Default is 'Process_ID', where ID is an
    integer value that is determined automatically.
    log_config: Configuration options for logging.

    InPorts
    -------
    added_input: The addition of all inputs (per dynamical system) at this
    timestep will be received by this port.
    replace_assignment: Todo: deprecate

    OutPorts
    --------
    messages: The payload to be sent to other dynamical systems when firing.
    local_cost: the cost component corresponding to this dynamical system, i.e.,
    c_i = sum_j{Q_{ij} \cdot x_i}  will be sent through this port. The cost
    integrator will then complete the cost computation  by adding all
    contributions, i.e., x^T \cdot Q \cdot x = sum_i{c_i}.

    """

    def __init__(self, *,
                 step_size: npt.ArrayLike,
                 shape: ty.Tuple[int, ...] = (1,),
                 init_state: npt.ArrayLike = 0,
                 noise_amplitude: npt.ArrayLike = 0,
                 input_duration: npt.ArrayLike = 6,
                 min_state: npt.ArrayLike = 1000,
                 min_integration: npt.ArrayLike = -1000,
                 steps_to_fire: npt.ArrayLike = 10,
                 refractory_period: npt.ArrayLike = 1,
                 cost_diagonal: npt.ArrayLike = 0,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None,
                 init_value: None = 0) -> None:
        super().__init__(
            shape=shape,
            initial_state=init_state,
            step_size=step_size,
            noise_amplitude=noise_amplitude,
            input_duration=input_duration,
            min_state=min_state,
            min_integration=min_integration,
            steps_to_fire=steps_to_fire,
            refractory_period=refractory_period,
            cost_diagonal=cost_diagonal,
            name=name,
            log_config=log_config,
            init_value=init_value
        )
        self.added_input = InPort(shape=shape)
        self.messages = OutPort(shape=shape)
        self.local_cost = OutPort(shape=shape)

        self.integration = Var(shape=shape, init=0)
        self.step_size = Var(shape=shape, init=step_size)
        self.state = Var(shape=shape, init=init_state)
        self.noise_amplitude = Var(shape=shape, init=noise_amplitude)
        self.input_duration = Var(shape=shape, init=input_duration)
        self.min_state = Var(shape=shape, init=min_state)
        self.min_integration = Var(shape=shape, init=min_integration)
        self.steps_to_fire = Var(shape=shape, init=steps_to_fire)
        self.refractory_period = Var(shape=shape, init=refractory_period)
        self.firing = Var(shape=shape, init=init_value)
        self.prev_assignment = Var(shape=shape, init=False)
        self.cost_diagonal = Var(shape=shape, init=cost_diagonal)
        self.assignment = Var(shape=shape, init=False)
        self.min_cost = Var(shape=shape, init=False)
