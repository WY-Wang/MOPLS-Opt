import sys
import logging
import time
import numpy as np
from copy import deepcopy
import scipy.spatial as scp
from matplotlib import pyplot as plt

from poap.strategy import Proposal, BaseStrategy, RetryStrategy
from pySOT.utils import from_unit_box, round_vars

from archiving_strategies import Memory_Record, Memory_Archive
from multiobjective_utilities import ND_add

logger = logging.getLogger(__name__)


class Multiobjective_Async_without_Constriants(BaseStrategy):

    # =====================================================================
    # Todo: Parameters initialization and preparation

    def __init__(self, mo_problem, surrogate, maxeval, nsamples, exp_design=None, sampling_method=None, extra=None, extra_vals=None, maxtime=np.inf, asynchronous=True):
        self.start_time = time.time()
        self.end_time = time.time()
        self.overhead = 0.0
        self.time_list = []
        self.time_list_nonoverhead = []

        self.maxeval = maxeval
        self.maxtime = maxtime

        self.mo_problem = mo_problem
        self.fhat = []
        for i in range(self.mo_problem.nobj):
            surrogate.reset()
            self.fhat.append(deepcopy(surrogate))

        self.ncenters = nsamples  # number of center points or number of threads for synchronization
        self.nsamples = 1  # number of sample points generated in the neighborhood of each center point
        self.numinit = None

        self.asynchronous = asynchronous
        self.permission_sampling = True  # ensure that no collision happens during adapt sampling
        self.permission_update = True  # ensure that no collision happens during updating
        self.batch_queue = []
        self.sample_preparation = []
        self.init_pending = 0
        self.pending_evals = 0
        self.xpend = np.empty([0, mo_problem.dim])
        self.phase = 1

        self.proposal_counter = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.terminate = False

        self.extra = extra
        self.extra_vals = extra_vals
        self.design = exp_design

        self.xrange = np.asarray(mo_problem.ub - mo_problem.lb)

        self.num_evals = 0
        self.init_evals = 0
        self.status = 0
        self.sigma = 0
        self.sigma_init = 0.2

        self.xbest = None
        self.fbest = np.inf
        self.fbest_old = None

        self.centers = []
        self.nd_archives = []
        self.memory_archive = Memory_Archive(300)
        self.tabu_list = []
        self.evals = []
        self.maxfit = 500
        self.d_thresh = 1.0
        self.sampling = sampling_method

        self.initial_preparation()

    def initial_preparation(self):
        # print('=== #1  sample initial ===')
        if self.num_evals == 0:
            logger.info("== Start ===")
        else:
            logger.info("=== Restart ===")

        for fhat in self.fhat:
            fhat.reset()
        self.sigma = self.sigma_init
        self.status = 0
        self.xbest = None
        self.fbest_old = None
        self.fbest = np.inf

        start_sample = self.design.generate_points()
        start_sample = from_unit_box(start_sample, self.mo_problem.lb, self.mo_problem.ub)

        if self.batch_queue is None:
            self.batch_queue = []

        if self.extra is not None:
            if self.num_evals > 0:
                for i in range(len(self.extra_vals)):
                    xx = self.projection(np.copy(self.extra[i, :]))
                    for j in range(self.mo_problem.nobj):
                        self.fhat[j].add_points(np.ravel(xx), self.extra_vals[i][j])
            else:
                if self.extra_vals is None:
                    self.extra_vals = np.nan * np.ones((self.extra.shape[0], self.mo_problem.nobj))

                for i in range(len(self.extra_vals)):
                    xx = self.projection(np.copy(self.extra[i, :]))
                    if np.isnan(self.extra_vals[i][0]) or np.isinf(self.extra_vals[i][0]):
                        self.batch_queue.append(np.ravel(xx))
                    else:
                        for j in range(self.mo_problem.nobj):
                            self.fhat[j].add_points(np.ravel(xx), self.extra_vals[i][j])

        for j in range(min(start_sample.shape[0], self.maxeval - self.num_evals)):
            start_sample[j, :] = self.projection(start_sample[j, :])
            self.sample_preparation.append(np.copy(start_sample[j, :]))
            self.init_evals += 1

        if self.extra is not None:
            self.sampling.init(np.vstack((start_sample, self.extra)), self.fhat, self.maxeval - self.num_evals)
        else:
            self.sampling.init(start_sample, self.fhat, self.maxeval - self.num_evals)

        if self.numinit is None:
            self.numinit = start_sample.shape[0]

    # =====================================================================
    # Todo: Control center for parallelization and proposal

    def propose_action(self):
        # print('phase = {}'.format(self.phase))
        # print('batch_queue = {}'.format(len(self.batch_queue)))
        # print('num_evals = {}, init_evals = {}'.format(self.num_evals, self.init_evals))

        if self.terminate:
            if self.pending_evals == 0:
                return Proposal('terminate')
        elif self.num_evals + self.pending_evals >= self.maxeval or self.terminate or self.end_time - self.start_time > self.maxtime:
            if self.pending_evals == 0:
                X = np.zeros((self.num_evals, self.mo_problem.dim + self.mo_problem.nobj + 2))
                all_xvals = np.asarray([rec.x for rec in self.evals])
                X[:, 0:self.mo_problem.dim] = all_xvals[:, :]
                all_fvals = np.asarray([rec.fx for rec in self.evals])
                X[:, self.mo_problem.dim:self.mo_problem.dim + self.mo_problem.nobj] = all_fvals[:, :]
                X[:, self.mo_problem.dim + self.mo_problem.nobj] = self.time_list[:]
                X[:, self.mo_problem.dim + self.mo_problem.nobj + 1] = self.time_list_nonoverhead[:]
                np.savetxt('final.txt', X)
                print('Normal Termination')
                return Proposal('terminate')
        elif self.batch_queue:
            if self.phase == 1:
                return self.init_proposal()
            else:
                return self.adapt_proposal()
        elif self.num_evals + self.pending_evals < self.init_evals or len(self.sample_preparation) > 0:
            if self.asynchronous:
                self.sample_init(ncenters=1)
            elif self.pending_evals == 0:
                self.sample_init(ncenters=self.ncenters)

            if self.terminate:
                if self.pending_evals == 0:
                    return Proposal('terminate')

            return self.init_proposal()
        elif self.num_evals >= self.init_evals:
            self.phase = 2
            if self.asynchronous:
                if self.permission_sampling:
                    print("Multiple Sampling = {}".format(self.ncenters - self.pending_evals))
                    self.sample_adapt(ncenters=self.ncenters - self.pending_evals)
            elif self.pending_evals == 0:
                self.sample_adapt(ncenters=self.ncenters)

            if self.terminate:
                if self.pending_evals == 0:
                    return Proposal('terminate')
            return self.adapt_proposal()

    def make_proposal(self, x):
        proposal = Proposal('eval', x)
        self.pending_evals += 1
        self.xpend = np.vstack((self.xpend, np.copy(x)))
        return proposal

    def remove_pending(self, x):
        index = np.where((self.xpend == x).all(axis=1))[0]
        self.xpend = np.delete(self.xpend, index, axis=0)

    # =====================================================================
    # Todo: Processing in initial phase

    def sample_init(self, ncenters):
        for _ in range(ncenters):
            if not self.sample_preparation:
                break
            self.batch_queue.append(np.copy(self.sample_preparation[0]))
            self.sample_preparation.remove(self.sample_preparation[0])

    def init_proposal(self):
        if self.batch_queue:
            proposal = self.make_proposal(self.batch_queue.pop())
            proposal.add_callback(self.on_initial_proposal)
            return proposal

    def on_initial_proposal(self, proposal):
        if proposal.accepted:
            self.on_initial_accepted(proposal)
        else:
            self.on_initial_rejected(proposal)

    def on_initial_accepted(self, proposal):
        self.accepted_count += 1
        proposal.record.add_callback(self.on_initial_update)

        # print("Arrange work for Worker {}".format(proposal.record.worker.ID))

    def on_initial_rejected(self, proposal):
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = proposal.args[0]
        self.batch_queue.append(xx)
        self.xpend = np.vstack((self.xpend, np.copy(xx)))
        self.remove_pending(xx)

    def on_initial_update(self, record):
        if record.status == 'completed':
            if self.permission_update:
                # print('Worker {} is idle'.format(record.worker.ID))
                self.on_initial_completed(record)
            else:
                self.on_initial_update(record)
        elif record.is_done:
            self.on_initial_aborted(record)

    def on_initial_completed(self, record):
        self.permission_update = False

        self.num_evals += 1
        self.pending_evals -= 1

        record.feasible = True
        self.log_completion(record)

        for i, fhat in enumerate(self.fhat):
            fhat.add_points(np.copy(record.params[0]), record.value[i])

        srec = Memory_Record(np.copy(record.params[0]), record.value, self.sigma_init)

        self.evals.append(srec)
        self.end_time = time.time()
        self.time_list.append(self.end_time - self.start_time)
        self.time_list_nonoverhead.append(self.end_time - self.start_time - self.overhead)

        self.remove_pending(np.copy(record.params[0]))

        self.memory_archive.add(srec)
        front = self.memory_archive.contents[0]
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)
        self.nd_archives = np.copy(fvals)
        self.d_thresh = 1 - float(self.num_evals - self.numinit) / float(self.maxeval - self.numinit)

        self.permission_update = True
        print(f'number of evaluations = {self.num_evals} after {self.end_time - self.start_time}')
        sys.stdout.flush()

    def on_initial_aborted(self, record):
        self.pending_evals -= 1
        xx = record.params[0]
        self.batch_queue.append(xx)
        self.remove_pending(xx)

    # =====================================================================
    # Todo: Processing in adaptive phase

    def sample_adapt(self, ncenters):
        print('number of evaluations = {}, number of pending = {}, size of batch queue = {}'.format(self.num_evals, self.pending_evals, len(self.batch_queue)))
        self.permission_sampling = False

        start = time.time()

        front = self.memory_archive.contents[0]
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)

        # print(nsamples)
        temp_centers = self.memory_archive.select_center_population(ncenters, self.d_thresh)
        for center in temp_centers:
            self.centers.append(center)
        nsamples = min(len(self.evals), self.nsamples * len(temp_centers), self.maxeval - self.num_evals)

        self.interactive_plotting(fvals)

        # -----------------------------------------------------
        # Candidate Selection
        if nsamples > 0:
            j = 0
            new_points = np.zeros((nsamples, self.mo_problem.dim))

            for rec in temp_centers:
                if rec.generated == 0:
                    xcenter = np.copy(rec.x)
                    xsigma = rec.sigma
                    if self.fhat[0].num_pts >= self.maxfit:
                        self.fit_local_surrogate(xcenter)

                    # start_time = time.time()
                    new_points[j:j + self.nsamples, :] = self.sampling.make_points(npts=1, xbest=xcenter, sigma=xsigma, front=fvals, projection=self.projection)
                    # end_time = time.time()
                    # print('Make Points Time = {}'.format(end_time - start_time))

                    rec.offsprings = np.copy(new_points[j:j + self.nsamples, :])
                    rec.generated = self.nsamples
                    j = j + self.nsamples
                    if j >= nsamples:
                        break
            # -----------------------------------------------------

            for i in range(nsamples):
                self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

        end = time.time()
        self.overhead += end - start

        self.permission_sampling = True

    def adapt_proposal(self):
        if self.batch_queue:
            proposal = self.make_proposal(self.batch_queue.pop())
            proposal.add_callback(self.on_adapt_proposal)
            return proposal

    def on_adapt_proposal(self, proposal):
        if proposal.accepted:
            self.on_adapt_accept(proposal)
        else:
            self.on_adapt_reject(proposal)

    def on_adapt_accept(self, proposal):
        self.accepted_count += 1
        proposal.record.add_callback(self.on_adapt_update)

        # print("Arrange work for Worker {}".format(proposal.record.worker.ID))

    def on_adapt_reject(self, proposal):
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = np.copy(proposal.args[0])
        self.remove_pending(xx)
        # if not self.asynchronous:  # Add back to the queue in synchronous case
        self.batch_queue.append(xx)

    def on_adapt_update(self, record):
        if record.status == 'completed':
            if self.permission_update:
                # print('Worker {} is idle'.format(record.worker.ID))
                self.on_adapt_completed(record)
            else:
                self.on_adapt_update(record)
        elif record.is_done:
            self.on_adapt_aborted(record)

    def on_adapt_completed(self, record):
        self.permission_update = False

        self.num_evals += 1
        # print(self.num_evals)
        self.pending_evals -= 1

        record.feasible = True
        # self.log_completion(record)

        singular = False
        for rec in self.evals:
            if scp.distance.euclidean(np.copy(rec.x), np.copy(record.params[0])) < 1e-6:
                singular = True
                break

        if not singular:
            i = 0
            for fhat in self.fhat:
                fhat.add_points(np.copy(record.params[0]), record.value[i])
                i += 1

        srec = Memory_Record(np.copy(record.params[0]), record.value, self.sigma_init)
        self.evals.append(srec)
        self.end_time = time.time()
        self.time_list.append(self.end_time - self.start_time)
        self.time_list_nonoverhead.append(self.end_time - self.start_time - self.overhead)

        self.remove_pending(np.copy(record.params[0]))

        self.update_memory(np.copy(record.params[0]), record.value)

        self.memory_archive.add(srec)
        front = self.memory_archive.contents[0]
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)
        self.nd_archives = np.copy(fvals)
        self.d_thresh = 1 - float(self.num_evals - self.numinit) / float(self.maxeval - self.numinit)

        self.permission_update = True
        print(f'number of evaluations = {self.num_evals} after {self.end_time - self.start_time}')
        sys.stdout.flush()

    def on_adapt_aborted(self, record):
        self.pending_evals -= 1
        xx = record.params[0]
        self.remove_pending(xx)

    # =====================================================================
    # Todo: Auxiliary tools

    def fit_local_surrogate(self, xbest):
        all_fvals = [rec.fx for rec in self.evals]
        all_fvals = np.asarray(all_fvals)
        all_xvals = [rec.x for rec in self.evals]
        all_xvals = np.asarray(all_xvals)

        distances = scp.distance.cdist(np.atleast_2d(xbest), all_xvals)
        index = np.ravel(np.argsort(distances))

        j = 0
        for fhat in self.fhat:
            fhat.reset()
            for i in range(self.maxfit):
                x = np.ravel(all_xvals[index[i], 0:self.mo_problem.dim])
                fx = all_fvals[index[i], j]
                fhat.add_points(x, fx)
            j += 1

    def update_memory(self, x_new, fval_new):
        F = np.vstack((self.nd_archives, np.asarray(fval_new)))
        (l, M) = F.shape
        nondominated = list(range(l - 1))
        dominated = []
        (nondominated, dominated) = ND_add(np.transpose(F), nondominated, dominated)

        found = False
        for center in self.centers:
            for offspring in center.offsprings:
                if np.array_equal(np.copy(x_new), offspring):
                    if dominated and dominated[0] == l - 1:
                        center.nfail += 1
                        center.sigma = center.sigma / 2
                        if center.nfail > 3:
                            center.ntabu = 5
                            center.nfail = 0
                            center.sigma = self.sigma_init
                            self.tabu_list.append(center)
                    center.generated -= 1
                    if center.generated == 0:
                        self.centers.remove(center)
                    found = True
                    break
            if found:
                break

        if (self.num_evals - self.init_evals) % self.ncenters == 0:
            for rec in self.tabu_list:
                rec.ntabu -= 1
                if rec.ntabu == 0:
                    self.tabu_list.remove(rec)

    def log_completion(self, record):
        xstr = np.array_str(record.params[0], max_line_width = np.inf, precision = 5, suppress_small = True)
        fstr = np.array_str(record.value, max_line_width = np.inf, precision = 5, suppress_small = True)
        if record.feasible:
            logger.info("{} {} @ {}".format("True", fstr, xstr))
        else:
            logger.info("{} {} @ {}".format("False", fstr, xstr))

    def projection(self, x):
        x = np.atleast_2d(x)
        return round_vars(x, self.mo_problem.int_var, self.mo_problem.lb, self.mo_problem.ub)

    def interactive_plotting(self, fvals):
        maxgen = (self.maxeval - self.numinit) / (self.nsamples * self.ncenters)
        curgen = (self.num_evals - self.numinit) / (self.nsamples * self.ncenters) + 1
        cent_fvals = [rec.fx for rec in self.centers]
        cent_fvals = np.asarray(cent_fvals)

        all_fvals = [rec.fx for rec in self.evals]
        all_fvals = np.asarray(all_fvals)
        plt.plot(all_fvals[:, 0], all_fvals[:, 1], 'k+')
        plt.plot(cent_fvals[:, 0], cent_fvals[:, 1], 'ro', markersize = 10)
        plt.plot(fvals[:, 0], fvals[:, 1], 'b*')
        if self.tabu_list:
            tabu_fvals = [rec.fx for rec in self.tabu_list]
            tabu_fvals = np.asarray(tabu_fvals)
            plt.plot(tabu_fvals[:, 0], tabu_fvals[:, 1], 'ys')
        plt.draw()
        plt.show()

