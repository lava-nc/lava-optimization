# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import cv2
import matplotlib.pylab as plt
import numpy as np
import os
from scipy.optimize import OptimizeResult
from skopt import Optimizer, Space
import skopt.plots as skplots
from skopt.space import Categorical, Integer, Real
from typing import Union

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.bayesian.processes import (
    BayesianOptimizer
)


@implements(proc=BayesianOptimizer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyBayesianOptimizerModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the Bayesian Optimizer processes. For
    more information, please refer to bayesian/processes.py.
    """
    results_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    next_point_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    acq_func_config = LavaPyType(np.ndarray, np.ndarray)
    acq_opt_config = LavaPyType(np.ndarray, np.ndarray)
    search_space = LavaPyType(np.ndarray, np.ndarray)
    enable_plotting = LavaPyType(np.ndarray, np.ndarray)
    est_config = LavaPyType(np.ndarray, np.ndarray)
    ip_gen_config = LavaPyType(np.ndarray, np.ndarray)
    log_dir = LavaPyType(np.ndarray, np.ndarray)
    num_ips = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)
    seed = LavaPyType(int, int)

    initialized = LavaPyType(bool, bool)
    num_iterations = LavaPyType(int, int)
    frame_log = LavaPyType(np.ndarray, np.ndarray)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""

        if self.initialized:
            # receive a result vector from the black-box function
            result_vec: np.ndarray = self.results_in.recv()

            self.opt_result: OptimizeResult = self.process_result_vector(
                result_vec
            )

            if self.enable_plotting[0]:
                self.plot_results(self.opt_result)
        else:
            # initialize the search space from the standard Bayesian
            # optimization search space schema; for more information,
            # please refer to the init_search_space method
            self.search_space = self.init_search_space()
            self.optimizer = Optimizer(
                dimensions=self.search_space,
                base_estimator=self.est_config[0],
                n_initial_points=self.num_ips,
                initial_point_generator=self.ip_gen_config[0],
                acq_func=self.acq_func_config[0],
                acq_optimizer=self.acq_opt_config[0],
                random_state=self.seed
            )
            self.frame_log[0]: dict = {
                'convergence': [],
                'evaluations': [],
                'gaussian_process': [],
                'objective': []
            }
            self.initialized: bool = True
            self.num_iterations: int = -1

        next_point: list = self.optimizer.ask()
        next_point: np.ndarray = np.ndarray(
            shape=(len(self.search_space), 1),
            buffer=np.array(next_point)
        )

        self.next_point_out.send(next_point)
        self.num_iterations += 1

    def __del__(self) -> None:
        """finalize the optimization processing upon runtime conclusion"""
        for _, frame_paths in self.frame_log[0].items():
            # calculate the maximum height and width of all images
            max_height, max_width = 0, 0
            for path in frame_paths:
                image = cv2.imread(path)
                height, width = image.shape[:2]
                max_height = max(max_height, height)
                max_width = max(max_width, width)

            # resize all images to the maximum size
            for path in frame_paths:
                image = cv2.imread(path)
                current_height, current_width = image.shape[:2]

                vertical_diff: int = max_height - current_height
                padding_top: int = vertical_diff // 2
                padding_bottom: int = vertical_diff - padding_top

                horizontal_diff: int = max_width - current_width
                padding_left: int = horizontal_diff // 2
                padding_right: int = horizontal_diff - padding_left

                padded_image = cv2.copyMakeBorder(
                    image,
                    padding_top,
                    padding_bottom,
                    padding_left,
                    padding_right,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )

                cv2.imwrite(path, padded_image)

        self.create_videos()

        if hasattr(self, "opt_result"):
            print(self.opt_result)

    def init_search_space(self) -> list[Space]:
        """initialize the search space from the standard schema

        This method is designed to convert the numpy ndarray-based search
        space description int scikit-optimize format compatible with all
        lower-level processes. Your search space should consist of three
        types of parameters:
            1) ("continuous", <min_value>, <max_value>, np.nan, <name>)
            2) ("integer", <min_value>, <max_value>, np.nan, <name>)
            3) ("categorical", np.nan, np.nan, <choices>, <name>)

        Returns
        -------
        search_space : list[Union[Real, Integer]]
            a collection of continuous and discrete dimensions that represent
            the entirety of the problem search space
        """
        search_space: list[Union[Real, Integer]] = []
        valid_param_types: list[str] = ["continuous", "integer", "categorical"]

        for i in range(self.search_space.shape[0]):
            p_type: str = self.search_space[i, 0]
            minimum: Union[int, float] = self.search_space[i, 1]
            maximum: Union[int, float] = self.search_space[i, 2]
            choices: list = self.search_space[i, 3]
            name: str = self.search_space[i, 4]

            if p_type == valid_param_types[0]:
                dimension = Real(
                    low=minimum,
                    high=maximum,
                    name=name
                )
            elif p_type == valid_param_types[1]:
                dimension = Integer(
                    low=minimum,
                    high=maximum,
                    name=name
                )
            elif p_type == valid_param_types[2]:
                dimension = Categorical(
                    categories=choices,
                    name=name
                )
            else:
                raise ValueError(
                    f"parameter type [{p_type}] is not in valid "
                    + f"parameter types: {valid_param_types}"
                )

            search_space.append(dimension)

        if not len(search_space) > 0:
            raise ValueError("search space is empty")

        return search_space

    def plot_results(self, results: OptimizeResult, dpi: int = 1000) -> None:
        """plot results from the latest surrogate model fitting

        Parameters
        ----------
        results : OptimizeResult
            information regarding the surrogate model's adaptation to the
            latest posterior information
        dpi : int; default = 1000
            specify the resolution of the plots
        """
        os.makedirs(self.log_dir[0], exist_ok=True)

        if len(results.models) > 0:
            # plot 1D Gaussian uncertainty for 1D optimization problems
            if len(self.search_space) == 1 and len(results.models) > 0:
                save_path: str = os.path.join(
                    self.log_dir[0],
                    f"gp_iter{self.num_iterations}.png"
                )
                skplots.plot_gaussian_process(results)
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                self.frame_log[0]['gaussian_process'].append(save_path)

            save_path: str = os.path.join(
                self.log_dir[0],
                f"obj_iter{self.num_iterations}.png"
            )
            skplots.plot_objective(results)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            self.frame_log[0]['objective'].append(save_path)

        save_path: str = os.path.join(
            self.log_dir[0],
            f"eval_iter{self.num_iterations}.png"
        )
        skplots.plot_evaluations(results)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        self.frame_log[0]['evaluations'].append(save_path)

        save_path: str = os.path.join(
            self.log_dir[0],
            f"convergence_iter{self.num_iterations}.png"
        )
        skplots.plot_convergence(results)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        self.frame_log[0]['convergence'].append(save_path)

    def create_videos(self, fps: int = 4) -> None:
        """compile videos of plot progression throughout modeling process

        Parameters
        ----------
        fps : int
            specify the frame rate of the compiled videos showing plot
            progression
        """

        # compile videos presenting the progression of various frames
        # throughout the optimization process
        for frame_type, frame_paths in self.frame_log[0].items():
            if len(frame_paths) > 0:
                first_frame = cv2.imread(frame_paths[0])
                height, width = first_frame.shape[:2]
                path: str = os.path.join(self.log_dir[0], f'{frame_type}.mp4')

                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

                for path in frame_paths:
                    frame = cv2.imread(path)
                    writer.write(frame)

                writer.release()

    def process_result_vector(self, vec: np.ndarray) -> None:
        """parse vec into params/objectives before informing optimizer

        Parameters
        ----------
        vec : np.ndarray
            a single array of data from the black-box process containing
            all parameters and objectives for a total length of num_params
            + num_objectives
        """
        vec: list = vec[:, 0].tolist()

        evaluated_point: list = vec[:-self.num_objectives]
        performance: list = vec[-self.num_objectives:]

        return self.optimizer.tell(evaluated_point, performance[0])
