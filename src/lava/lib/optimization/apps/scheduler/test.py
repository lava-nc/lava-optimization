from lava.lib.optimization.apps.scheduler.problems import (
    SatelliteScheduleProblem)
from lava.lib.optimization.apps.scheduler.solver import SatelliteScheduler

if __name__ == "__main__":
    ssp = SatelliteScheduleProblem(num_satellites=12,
                                   num_requests=100)
    ssp.generate(42)
    # ssp.plot_problem()

    sat_scheduler = SatelliteScheduler(ssp,
                                       qubo_weights=(4, 20))

    sat_scheduler.solve_with_netx()
    print(f'Scheduled {sat_scheduler.netx_solution.shape[0]} Requests.')

    sat_scheduler.qubo_hyperparams = ({}, False)
    sat_scheduler.lava_backend = "Loihi2"
    sat_scheduler.solve_with_lava_qubo(timeout=100000)

    print(f'Scheduled {sat_scheduler.lava_solution.shape[0]} Requests.')


