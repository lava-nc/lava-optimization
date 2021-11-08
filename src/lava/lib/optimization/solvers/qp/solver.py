# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from src.lava.lib.optimization.solvers.abstract.solver import OptimizationSolver


class QPSolver(OptimizationSolver):

    def create_network(self, problem):
        return None

    def create_integrator(self, problem):
        return None

    def _run(self, timeout, backend=None):
        self.solver_net.run()

    def solve(self, problem, **kwargs):
        self.solver_net = self.create_network(problem)
        self.integrator = self.create_integrator(problem)
        self._build()
        super().solve(**kwargs)
