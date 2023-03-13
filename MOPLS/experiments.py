import os.path
import time
import numpy as np

from poap.controller import ThreadController, BasicWorkerThread
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.experimental_design import SymmetricLatinHypercube

from multiobjective_problems import DTLZ2
from multiobjective_sampling_strategies import CandidateDYCORS
from multiobjective_strategies import Multiobjective_Async_without_Constriants


def main():
    delay = False
    asynchronous = False

    nthreads = 4
    maxeval = 600
    maxtime = 7200

    for nobj in [2]:
        for dim in [10]:
            for mo_problem in [
                DTLZ2(dim=dim, nobj=nobj, delay=delay, alpha=102.0, xm=1.0),
            ]:
                for trial in range(20):
                    print(f"Problem being solved: {mo_problem.name}")
                    print(f"Number of Threads: {nthreads}")
                    print(f"Trial Number: {trial}")

                    single_experiment(mo_problem, nthreads, maxeval, maxtime, trial, asynchronous)


def single_experiment(mo_problem, nthreads, maxeval, maxtime, trial_number, asynchronous):
    np.random.seed(trial_number)

    nsamples = nthreads
    ncands = 100 * mo_problem.dim

    surrogate = RBFInterpolant(
        dim=mo_problem.dim,
        lb=mo_problem.lb,
        ub=mo_problem.ub,
        kernel=CubicKernel(),
        tail=LinearTail(mo_problem.dim),
    )
    exp_design = SymmetricLatinHypercube(dim=mo_problem.dim, num_pts=2 * (mo_problem.dim + 1))

    controller = ThreadController()
    controller.strategy = Multiobjective_Async_without_Constriants(
        mo_problem=mo_problem,
        maxeval=maxeval,
        maxtime=maxtime,
        nsamples=nsamples,
        exp_design=exp_design,
        surrogate=surrogate,
        sampling_method=CandidateDYCORS(mo_problem=mo_problem, ncands=ncands),
        asynchronous=asynchronous,
    )

    for worker_ID in range(nsamples):
        worker = BasicWorkerThread(controller, mo_problem.eval)
        worker.ID = worker_ID
        controller.launch_worker(worker)

    def merit(r):
        return r.value[0]

    start = time.time()
    controller.run(merit=merit)
    end = time.time()
    print(f'time = {end - start}')

    result = np.loadtxt('final.txt')
    # controller.strategy.save_plot(trial_number)

    fname = f"{mo_problem.name}_{mo_problem.nobj}_{mo_problem.dim}_MOPLS_{maxeval}_{trial_number}_{nthreads}.txt"
    np.savetxt(fname, result)


if __name__ == '__main__':
    main()