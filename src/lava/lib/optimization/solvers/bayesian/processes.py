import numpy as np

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var

class BayesianOptimizer(AbstractProcess):
    """
    An abstract process defining the internal state and input/output
    variables required by all Bayesian optimizers
    """

    def __init__(self, acq_func_config: dict, acq_opt_config: dict,
            search_space: np.ndarray, enable_plotting: bool, est_config: str,
            ip_gen_config: dict, num_ips: int, num_objectives: int,
            log_dir: str, seed: int, **kwargs) -> None:
        """initialize the BayesianOptimizer process

        Parameters
        ----------
        acq_func_config : dict
            a series of key-value pairs specifying the runtime configuration
            of the acquisition function
        acq_opt_config : dict
            a series of key-value pairs specifying the runtime configuration
            of the acquisition optimizer
        search_space : np.ndarray
            A list of parameters that describe the multi-dimensional search
            space of the Bayesian optimization process. The length of this
            data structure represents the number of parameters in the search
            space with object at each index ( dimensions[idx] ) describing
            the range or discrete choices for the idx-th parameters. Your
            search space should consist of two types of parameters:
                1) continuous: (<min_value>, <max_value>, np.inf)
                2) discrete: (<min_value>, <max_value>, <float>)

            The following represents the three main types of parameters:
                1) a continuous variable from -1000.0 to 256
                3) a discrete set of categorical variable, the best way to
                    convert your categorical variables to this format is to
                    exploit the discretization of enumerations or parse your
                    list possible selections using array indexing.

            search_space: list = np.ndarray([
                [-1000, 256, np.inf],
                [0, <num_categories>, 1], 
            ])
        enable_plotting : bool
            a boolean specifying whether or not to generate plots of the
            regressors current state at the end of every iteration
        est_config : dict
            a series of key-value pairs specifying the runtime configuration
            of the surrogate function estimator
        ip_gen_config : dict
            a series of key-value pairs specifying the runtime configuration
            of the initial point generator
        num_ips : int
            an integer specifying the number of points to sample from the
            parameter search space before using the surrogate gradients to
            exploit the search space
        num_objectives : int
            an integer specifying the number of qualitative attributes used
            to measure the black-box function
        log_dir : str
            a string specifying the directory where all runtime logs shall
            be saved
        seed : int
            an integer specifying the random state of all random number
            generators
        """
        super().__init__(**kwargs)

        num_ss_dimensions: int = search_space.shape[0]

        # Input/Output Ports
        input_length: int = num_ss_dimensions + num_objectives
        output_length: int = num_ss_dimensions
        self.results_in = InPort((input_length, 1))
        self.next_point_out = OutPort((output_length, 1))

        # General Internal State Variables
        sorted_keys: list[str] = sorted(acq_func_config.keys())
        sorted_config: list = [acq_func_config[k] for k in sorted_keys]
        sorted_config: np.ndarray = np.array(sorted_config)
        self.acq_func_config = Var(
            shape=sorted_config.shape,
            init=sorted_config
        )

        sorted_keys: list[str] = sorted(acq_func_config.keys())
        sorted_config: list = [acq_opt_config[k] for k in sorted_keys]
        sorted_config: np.ndarray = np.array(sorted_config)
        self.acq_opt_config = Var(
            shape=sorted_config.shape,
            init=sorted_config
        )

        sorted_keys: list[str] = sorted(est_config.keys())
        sorted_config: list = [est_config[k] for k in sorted_keys]
        sorted_config: np.ndarray = np.array(sorted_config)
        self.est_config = Var(shape=sorted_config.shape, init=sorted_config)

        sorted_keys: list[str] = sorted(ip_gen_config.keys())
        sorted_config: list = [ip_gen_config[k] for k in sorted_keys]
        sorted_config: np.ndarray = np.array(sorted_config)
        self.ip_gen_config = Var(
            shape=sorted_config.shape,
            init=sorted_config
        )
        
        self.search_space = Var(search_space.shape, init=search_space)
        self.enable_plotting = Var((1,), init=np.array([enable_plotting]))
        self.log_dir = Var((1,), init=np.array([log_dir]))
        self.num_ips = Var((1,), init=num_ips)
        self.num_objectives = Var((1,), init=num_objectives)
        self.seed = Var((1,), init=seed)

        self.frame_log = Var((1,), init=np.array([{}]))
        self.num_iterations = Var((1,), init=-1)
        self.initialized = Var((1,), init=False)
