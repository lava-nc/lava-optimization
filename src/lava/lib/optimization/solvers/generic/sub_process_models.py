# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:

    class NetL2:
        pass


import numpy as np
from lava.lib.optimization.solvers.generic.cost_integrator.process import (
    CostIntegrator,
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    CostConvergenceChecker,
    DiscreteVariablesProcess,
    ContinuousVariablesProcess,
    ContinuousConstraintsProcess,
    StochasticIntegrateAndFire,
    NEBMAbstract,
    SimulatedAnnealingAbstract,
    SimulatedAnnealingLocalAbstract,
)
from lava.magma.core.resources import (
    CPU,
    Loihi2NeuroCore,
    NeuroCore,
)
from lava.lib.optimization.solvers.generic.nebm.process import (
    NEBM,
    SimulatedAnnealing,
    SimulatedAnnealingLocal,
)
from lava.lib.optimization.solvers.generic.annealing.process import Annealing
from lava.lib.optimization.solvers.generic.scif.process import QuboScif
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.lib.optimization.solvers.generic.qp.models import (
    ProjectedGradientNeuronsPIPGeq,
    ProportionalIntegralNeuronsPIPGeq,
)
from lava.proc.sparse.process import Sparse
from lava.lib.optimization.utils.datatype_converter import convert_to_fp
from scipy.sparse import csr_matrix

CPUS = [CPU, "CPU"]
NEUROCORES = [Loihi2NeuroCore, NeuroCore, "Loihi2"]
BACKEND_MSG = f""" was requested as backend. However,
the solver currently supports only Loihi 2 and CPU backends.
These can be specified by calling solve with any of the following:
backend = "CPU"
backend = "Loihi2"
backend = CPU
backend = Loihi2NeuroCore
backend = NeuroCoreS
The explicit resource classes can be imported from
lava.magma.core.resources"""


@implements(proc=ContinuousVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class ContinuousVariablesModel(AbstractSubProcessModel):
    """Model for the ContinuousVariables process."""

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        backend = proc.backend
        neuron_model = proc.hyperparameters.get("neuron_model", "qp-lp_pipg")

        if neuron_model == "qp-lp_pipg":
            # adding them here to show that they are need for the neurons models
            # since some values are calculated based on these weights
            Q_pre = proc.problem.hessian
            A_pre = proc.problem.constraint_hyperplanes_eq
            p_pre = proc.problem.linear_offset
            # legitimate hyperparameters
            init_state = proc.hyperparameters.get(
                "init_state", np.zeros((p_pre.shape[0],), dtype=int)
            )
            lr_change = proc.hyperparameters.get("lr_change_type", "indices")
            alpha_man = proc.hyperparameters.get("alpha_mantissa", 1)
            alpha_exp = proc.hyperparameters.get("alpha_exponent", 1)
            decay_params = proc.hyperparameters.get(
                "decay_schedule_parameters", (100, 100, 0)
            )
            alpha_decay_indices = proc.hyperparameters.get(
                "alpha_decay_indices", [0]
            )
            alpha = proc.hyperparameters.get("alpha", 1)
            if backend in CPUS:
                self.ProjGrad = ProjectedGradientNeuronsPIPGeq(
                    shape=init_state.shape,
                    qp_neurons_init=init_state,
                    grad_bias=p_pre,
                    alpha=alpha,
                    lr_decay_type=lr_change,
                    alpha_decay_params=decay_params,
                    alpha_decay_indices=alpha_decay_indices,
                )
            elif backend in NEUROCORES:
                _, Q_pre_fp_exp = convert_to_fp(Q_pre, 8)
                _, A_pre_fp_exp = convert_to_fp(A_pre, 8)
                p_pre_fp_man, p_pre_fp_exp = convert_to_fp(p_pre, 24)
                correction_exp = min(A_pre_fp_exp, Q_pre_fp_exp)

                self.ProjGrad = ProjectedGradientNeuronsPIPGeq(
                    shape=init_state.shape,
                    qp_neurons_init=init_state,
                    da_exp=correction_exp,
                    grad_bias=p_pre_fp_man,
                    grad_bias_exp=p_pre_fp_exp,
                    alpha=alpha_man,
                    alpha_exp=alpha_exp,
                    alpha_decay_params=decay_params,
                )
            else:
                raise NotImplementedError(str(backend) + BACKEND_MSG)

            # Connect the parent InPort to InPort of the Dense child-Process.
            proc.in_ports.a_in.connect(self.ProjGrad.in_ports.a_in)
            self.ProjGrad.out_ports.s_out.connect(proc.out_ports.s_out)
            proc.vars.variable_assignment.alias(self.ProjGrad.qp_neuron_state)
        else:
            AssertionError("Unknown neuron model specified")


@implements(proc=ContinuousConstraintsProcess, protocol=LoihiProtocol)
@requires(CPU)
class ContinuousConstraintsModel(AbstractSubProcessModel):
    """Model for the ContinuousConstraints process."""

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        backend = proc.backend
        neuron_model = proc.hyperparameters.get("neuron_model", "qp-lp_pipg")

        if neuron_model == "qp-lp_pipg":
            # adding them here to show that they are need for the neurons models
            # since some values are calculated based on these weights
            Q_pre = proc.problem.hessian
            A_pre = proc.problem.constraint_hyperplanes_eq
            k_pre = proc.problem.constraint_biases_eq
            # legitimate hyperparameters
            init_constraints = proc.hyperparameters.get(
                "init_constraints", np.zeros((k_pre.shape[0],), dtype=int)
            )
            beta_man = proc.hyperparameters.get("beta_mantissa", 1)
            beta_exp = proc.hyperparameters.get("beta_exponent", 1)
            beta = proc.hyperparameters.get("beta", 1)
            growth_params = proc.hyperparameters.get(
                "growth_schedule_parameters", (3, 2)
            )
            beta_growth_indices = proc.hyperparameters.get(
                "beta_growth_indices", [0]
            )
            lr_change = proc.hyperparameters.get("lr_change_type", "indices")

            if backend in CPUS:
                self.conn_A = Sparse(
                    weights=csr_matrix(A_pre),
                    num_message_bits=64,
                )

                self.conn_A_T = Sparse(
                    weights=csr_matrix(A_pre.T), num_message_bits=64
                )

                # Neurons for Constraint Checking
                self.ProInt = ProportionalIntegralNeuronsPIPGeq(
                    shape=init_constraints.shape,
                    constraint_neurons_init=init_constraints,
                    thresholds=k_pre,
                    beta=beta,
                    lr_growth_type=lr_change,
                    beta_growth_params=growth_params,
                    beta_growth_indices=beta_growth_indices,
                )
            elif backend in NEUROCORES:
                _, Q_pre_fp_exp = convert_to_fp(Q_pre, 8)
                A_pre_fp_man, A_pre_fp_exp = convert_to_fp(A_pre, 8)
                k_pre_fp_man, k_pre_fp_exp = convert_to_fp(k_pre, 24)
                correction_exp = min(A_pre_fp_exp, Q_pre_fp_exp)
                A_exp_new = -correction_exp + A_pre_fp_exp
                A_pre_fp_man = (A_pre_fp_man // 2) * 2

                self.conn_A = Sparse(
                    weights=csr_matrix(A_pre_fp_man),
                    num_message_bits=24,
                )

                self.conn_A_T = Sparse(
                    weights=csr_matrix(A_pre_fp_man.T),
                    weight_exp=A_exp_new,
                    num_message_bits=24,
                )

                # Neurons for Constraint Checking
                self.ProInt = ProportionalIntegralNeuronsPIPGeq(
                    shape=init_constraints.shape,
                    constraint_neurons_init=init_constraints,
                    da_exp=A_pre_fp_exp,
                    thresholds=k_pre_fp_man,
                    thresholds_exp=k_pre_fp_exp,
                    beta=beta_man,
                    beta_exp=beta_exp,
                    beta_growth_params=growth_params,
                )
            else:
                raise NotImplementedError(str(backend) + BACKEND_MSG)

            proc.in_ports.a_in.connect(self.conn_A.s_in)
            self.conn_A.a_out.connect(self.ProInt.a_in)
            self.ProInt.s_out.connect(self.conn_A_T.s_in)
            self.conn_A_T.a_out.connect(proc.out_ports.s_out)
            proc.vars.constraint_assignment.alias(
                self.ProInt.constraint_neuron_state
            )
        else:
            AssertionError("Unknown neuron model specified")


@implements(proc=DiscreteVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class DiscreteVariablesModel(AbstractSubProcessModel):
    """Model for the DiscreteVariables process.

    The model composes a population of Boltzmann units and
    connects them via Dense processes as to represent integer or binary
    variables.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        wta_weight = -2
        shape = proc.proc_params.get("shape", (1,))
        diagonal = proc.proc_params.get("cost_diagonal")
        weights = proc.proc_params.get(
            "weights",
            wta_weight
            * np.logical_not(np.eye(shape[1] if len(shape) == 2 else 0)),
        )
        neuron_model = proc.hyperparameters.get("neuron_model")

        cost_off_diagonal = proc.proc_params.get("cost_off_diagonal")

        if neuron_model in ["sa", "nebm-sa-refract"]:
            max_temperature = proc.hyperparameters.get("max_temperature", 1000)
            min_temperature = proc.hyperparameters.get("min_temperature", 0)
            delta_temperature = proc.hyperparameters.get("delta_temperature", 1)
            exp_temperature = proc.hyperparameters.get("exp_temperature", None)
            steps_per_temperature = proc.hyperparameters.get(
                "steps_per_temperature", 1
            )
            init_value = proc.hyperparameters.get(
                "init_value", np.zeros(shape, dtype=int))
            init_state = cost_off_diagonal @ init_value
            annealing_schedule = proc.hyperparameters.get("annealing_schedule",
                                                          'linear')
            nebm_params = {
                'shape': shape,
                'max_temperature': max_temperature,
                'min_temperature': min_temperature,
                'delta_temperature': delta_temperature,
                'exp_temperature': exp_temperature,
                'steps_per_temperature': steps_per_temperature,
                'cost_diagonal': diagonal,
                'init_value': init_value,
                'init_state': init_state,
                'annealing_schedule': annealing_schedule,
            }
            if neuron_model == 'sa':
                nebm_params['cost_off_diagonal'] = cost_off_diagonal
                self.s_bit = SimulatedAnnealingAbstract(
                    **nebm_params
                )
            elif neuron_model == 'nebm-sa-refract':
                refract_seed = proc.hyperparameters.get("refract_seed", 0)
                refract_scaling = proc.hyperparameters.get("refract_scaling", 4)
                nebm_params['refract_scaling'] = refract_scaling
                nebm_params['refract_seed'] = refract_seed
                self.s_bit = SimulatedAnnealingLocalAbstract(
                    **nebm_params
                )
        elif neuron_model == "nebm":
            raise NotImplementedError(
                "The neuron model nebm has been deprecated. Please use the "
                "more performant 'nebm-sa' neuron model.")
            temperature = proc.hyperparameters.get("temperature", 1)
            refract = proc.hyperparameters.get("refract", 0)
            refract_counter = proc.hyperparameters.get("refract_counter", 0)
            init_value = proc.hyperparameters.get(
                "init_value", np.zeros(shape, dtype=int)
            )
            init_state = proc.hyperparameters.get(
                "init_state", np.zeros(shape, dtype=int)
            )

            nebm_params = {
                'temperature': temperature,
                'refract': refract,
                'refract_counter': refract_counter,
                'init_state': init_state,
                'shape': shape,
                'cost_diagonal': diagonal,
                'init_value': init_value}

            self.s_bit = NEBMAbstract(**nebm_params)
        elif neuron_model == 'scif':
            noise_amplitude = proc.hyperparameters.get("noise_amplitude", 1)
            noise_precision = proc.hyperparameters.get("noise_precision", 5)
            init_value = proc.hyperparameters.get(
                "init_value", np.zeros(shape)
            )
            init_state = proc.hyperparameters.get(
                "init_value", np.zeros(shape)
            )
            on_tau = proc.hyperparameters.get("sustained_on_tau", (-3))

            self.s_bit = StochasticIntegrateAndFire(
                shape=shape,
                step_size=diagonal,
                init_state=init_state,
                init_value=init_value,
                noise_amplitude=noise_amplitude,
                noise_precision=noise_precision,
                sustained_on_tau=on_tau,
                cost_diagonal=diagonal,
            )
        else:
            AssertionError("Unknown neuron model specified")
        if weights.shape != (0, 0):
            self.sparse = Sparse(weights=weights)
            self.s_bit.out_ports.messages.connect(self.sparse.in_ports.s_in)
            self.sparse.out_ports.a_out.connect(
                self.s_bit.in_ports.added_input
            )

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.a_in.connect(self.s_bit.in_ports.added_input)
        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.s_bit.out_ports.messages.connect(proc.out_ports.s_out)
        self.s_bit.out_ports.local_cost.connect(proc.out_ports.local_cost)
        proc.vars.variable_assignment.alias(self.s_bit.prev_assignment)

    @staticmethod
    def get_neuron_process(hyperparameters):
        """Given the neuron_model, return the appropriate class for the
        neurons representing discrete variables."""

        neuron_model = hyperparameters.get("neuron_model")

        if neuron_model == 'sa':
            return SimulatedAnnealingAbstract
        elif neuron_model == 'nebm-sa-refract':
            return SimulatedAnnealingLocalAbstract
        elif neuron_model == 'nebm':
            return NEBMAbstract
        elif neuron_model == 'scif':
            return StochasticIntegrateAndFire
        else:
            raise ValueError("Please choose a supported neuron model")


@implements(proc=CostConvergenceChecker, protocol=LoihiProtocol)
@requires(CPU)
class CostConvergenceCheckerModel(AbstractSubProcessModel):
    """Model for the CostConvergence process.

    The model composes a CostIntegrator unit with incomming connections,
    in this way, downstream processes can be directly connected to the
    CostConvergence process.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        shape = proc.proc_params.get("shape", (1,))
        weights = proc.proc_params.get("weights", np.ones((1, shape[0])))
        self.dense = Dense(weights=weights, num_message_bits=24)
        self.cost_integrator = CostIntegrator(shape=(1,), min_cost=0)
        self.dense.out_ports.a_out.connect(
            self.cost_integrator.in_ports.cost_in
        )

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.cost_components.connect(self.dense.in_ports.s_in)

        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        # Communicates the last 3 bytes, and the first byte
        # Total cost = cost_min_first_byte << 24 + cost_min_last_bytes
        self.cost_integrator.out_ports.cost_out_last_bytes.connect(
            proc.out_ports.cost_out_last_bytes
        )
        self.cost_integrator.out_ports.cost_out_first_byte.connect(
            proc.out_ports.cost_out_first_byte
        )
        # Note: Total min cost = cost_min_first_byte << 24 + cost_min_last_bytes
        proc.vars.cost_min_last_bytes.alias(
            self.cost_integrator.vars.cost_min_last_bytes
        )
        proc.vars.cost_min_first_byte.alias(
            self.cost_integrator.vars.cost_min_first_byte
        )
        proc.vars.cost_last_bytes.alias(
            self.cost_integrator.vars.cost_last_bytes
        )
        proc.vars.cost_first_byte.alias(
            self.cost_integrator.vars.cost_first_byte
        )


@implements(proc=StochasticIntegrateAndFire, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class StochasticIntegrateAndFireModelSCIF(AbstractSubProcessModel):
    """Model for the StochasticIntegrateAndFire process.
    The process is just a wrapper over the QuboScif process.
    # Todo deprecate in favour of QuboScif.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        step_size = proc.proc_params.get("step_size", (1,))
        theta = proc.proc_params.get("threshold", (1,))
        cost_diagonal = proc.proc_params.get("cost_diagonal", (1,))
        noise_amplitude = proc.proc_params.get("noise_amplitude", (1,))
        noise_precision = proc.proc_params.get("noise_precision", (3,))
        sustained_on_tau = proc.proc_params.get("sustained_on_tau", (-5,))
        self.scif = QuboScif(
            shape=shape,
            cost_diag=cost_diagonal,
            theta=theta,
            sustained_on_tau=sustained_on_tau,
            noise_amplitude=noise_amplitude,
            noise_precision=noise_precision,
        )
        proc.in_ports.added_input.connect(self.scif.in_ports.a_in)
        self.scif.s_wta_out.connect(proc.out_ports.messages)
        self.scif.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.scif.vars.spk_hist)
        proc.vars.state.alias(self.scif.vars.state)
        proc.vars.cost_diagonal.alias(self.scif.vars.cost_diagonal)
        proc.vars.noise_amplitude.alias(self.scif.vars.noise_ampl)
        proc.vars.noise_precision.alias(self.scif.vars.noise_prec)


@implements(proc=NEBMAbstract, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NEBMAbstractModel(AbstractSubProcessModel):
    """
    ProcessModel for an NEBM process.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        temperature = proc.proc_params.get("temperature", (1,))
        refract = proc.proc_params.get("refract", (1,))
        refract_counter = proc.proc_params.get("refract_counter", (0,))
        init_value = proc.proc_params.get("init_value", np.zeros(shape))
        init_state = proc.proc_params.get("init_state", np.zeros(shape))
        self.nebm = NEBM(shape=shape,
                         temperature=temperature,
                         refract=refract,
                         refract_counter=refract_counter,
                         init_value=init_value,
                         init_state=init_state)
        proc.in_ports.added_input.connect(self.nebm.in_ports.a_in)
        self.nebm.s_wta_out.connect(proc.out_ports.messages)
        self.nebm.s_sig_out.connect(proc.out_ports.local_cost)
        proc.vars.prev_assignment.alias(self.nebm.vars.spk_hist)
        proc.vars.state.alias(self.nebm.vars.state)


@implements(proc=SimulatedAnnealingLocalAbstract, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class SimulatedAnnealingLocalAbstractModel(AbstractSubProcessModel):
    """
    ProcessModel for an NEBM process with Simulated Annealing.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        cost_diagonal = proc.proc_params.get("cost_diagonal")
        max_temperature = proc.proc_params.get("max_temperature")
        min_temperature = proc.proc_params.get("min_temperature")
        annealing_schedule = proc.proc_params.get("annealing_schedule")
        delta_temperature = proc.proc_params.get("delta_temperature")
        exp_temperature = proc.proc_params.get("exp_temperature")
        steps_per_temperature = proc.proc_params.get("steps_per_temperature")
        init_value = proc.proc_params.get("init_value", np.zeros(shape))
        init_state = proc.proc_params.get("init_state", np.zeros(shape))
        refract_scaling = proc.proc_params.get("refract_scaling")
        refract_seed = proc.proc_params.get("refract_seed")

        annealing_params = {
            'shape': (1,),
            'max_temperature': max_temperature,
            'min_temperature': min_temperature,
            'delta_temperature': delta_temperature,
            'steps_per_temperature': steps_per_temperature,
            'exp_temperature': exp_temperature,
            'annealing_schedule': annealing_schedule,
        }

        nebm_params = {
            'shape': shape,
            'max_temperature': max_temperature,
            'cost_diagonal': cost_diagonal,
            'init_value': init_value,
            'init_state': init_state,
            'refract_scaling': refract_scaling,
            'refract_seed': refract_seed}

        self.nebm = SimulatedAnnealingLocal(**nebm_params)

        self.annealing = Annealing(
            **annealing_params
        )
        # Connect Annealing neuron to NEBMSimulatedAnnealing neurons
        weights_anneal = np.ones((shape[0], 1))
        self.dense = Dense(weights=weights_anneal, num_message_bits=24)
        self.annealing.out_ports.delta_temperature_out.connect(
            self.dense.in_ports.s_in
        )
        self.dense.out_ports.a_out.connect(
            self.nebm.in_ports.delta_temperature_in
        )

        proc.in_ports.added_input.connect(self.nebm.in_ports.a_in)
        self.nebm.s_wta_out.connect(proc.out_ports.messages)
        self.nebm.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.nebm.vars.spk_hist)
        proc.vars.state.alias(self.nebm.vars.state)
        proc.vars.temperature.alias(self.annealing.temperature)


@implements(proc=SimulatedAnnealingAbstract, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class SimulatedAnnealingAbstractModel(AbstractSubProcessModel):
    """
    ProcessModel for an NEBM process with Simulated Annealing.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        cost_diagonal = proc.proc_params.get("cost_diagonal")
        max_temperature = proc.proc_params.get("max_temperature")
        min_temperature = proc.proc_params.get("min_temperature")
        annealing_schedule = proc.proc_params.get("annealing_schedule")
        delta_temperature = proc.proc_params.get("delta_temperature")
        exp_temperature = proc.proc_params.get("exp_temperature")
        steps_per_temperature = proc.proc_params.get("steps_per_temperature")
        init_value = proc.proc_params.get("init_value", np.zeros(shape))
        init_state = proc.proc_params.get("init_state", np.zeros(shape))
        cost_off_diagonal = proc.proc_params.get("cost_off_diagonal")

        annealing_params = {
            'shape': (1,),
            'max_temperature': max_temperature,
            'min_temperature': min_temperature,
            'delta_temperature': delta_temperature,
            'steps_per_temperature': steps_per_temperature,
            'exp_temperature': exp_temperature,
            'annealing_schedule': annealing_schedule,
        }

        nebm_params = {
            'shape': shape,
            'max_temperature': max_temperature,
            'cost_diagonal': cost_diagonal,
            'init_value': init_value,
            'init_state': init_state,
        }

        # the simulated annealing neuron requires 2 Loihi time steps for a
        # single algorithmic time step
        annealing_params['steps_per_temperature'] = 2 * steps_per_temperature
        self.nebm = SimulatedAnnealing(
            **nebm_params
        )

        self.annealing = Annealing(
            **annealing_params
        )
        # Connect Annealing neuron to NEBMSimulatedAnnealing neurons
        weights_anneal = np.ones((shape[0], 1))
        self.dense_annealing = Dense(weights=weights_anneal,
                                     num_message_bits=24)
        self.annealing.out_ports.delta_temperature_out.connect(
            self.dense_annealing.in_ports.s_in
        )
        self.dense_annealing.out_ports.a_out.connect(
            self.nebm.in_ports.delta_temperature_in
        )

        # Connect NEBM neurons to avoid conflicting variable switching
        # If Q_ij == 0, then neurons i and j can always independently switch
        # variables without conflict. Thus, only link neurons where Q_ij != 0
        # ToDo: Ensure that synapses are all 1bit
        weights_conflicts = (cost_off_diagonal != 0).astype(int)
        self.sparse_conflicts = Sparse(weights=weights_conflicts,
                                       num_message_bits=24)
        self.nebm.suggest_switch_out.connect(self.sparse_conflicts.s_in)
        self.sparse_conflicts.a_out.connect(self.nebm.n_switches_in)

        proc.in_ports.added_input.connect(self.nebm.in_ports.a_in)
        self.nebm.s_wta_out.connect(proc.out_ports.messages)
        self.nebm.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.nebm.vars.spk_hist)
        proc.vars.state.alias(self.nebm.vars.state)
        proc.vars.temperature.alias(self.annealing.temperature)
