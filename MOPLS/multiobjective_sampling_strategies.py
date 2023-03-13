import math
import random
import numpy as np
import scipy.stats as stats
import scipy.spatial as scp
# from hv import HyperVolume
from pyhv import HyperVolume
from multiobjective_utilities import ND_add, ND_front, normalize_objectives

INF = float('inf')

class Multiple_Sampling(object):
    def __init__(self, strategy_list, cycle):
        if cycle is None:
            cycle = list(range(len(strategy_list)))
        self.cycle = cycle
        self.strategy_list = strategy_list
        self.nstrategies = len(strategy_list)

        self.current_strategy = 0
        self.proposed_points = None
        self.mo_problem = strategy_list[0].mo_problem
        self.fhat = None
        self.budget = None
        self.n0 = None

    def init(self, start_sample, fhat, budget):
        self.proposed_points = start_sample
        self.fhat = fhat
        self.n0 = start_sample.shape[0]
        for i in range(self.nstrategies):
            self.strategy_list[i].init(self.proposed_points, fhat, budget)

    def remove_point(self, x):
        index = np.sum(np.abs(self.proposed_points - x), axis = 1).argmin()
        if np.sum(np.abs(self.proposed_points[index, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, index, axis = 0)
            for i in range(self.nstrategies):
                self.strategy_list[i].remove_point(x)
            return True
        return False

    def make_points(self, npts, xbest, sigma, front, subset = None, projection = None):
        new_points = np.zeros((npts, self.mo_problem.dim))

        npoints = np.zeros((self.nstrategies,), dtype = int)
        for i in range(npts):
            npoints[self.cycle[self.current_strategy]] += 1
            self.current_strategy = (self.current_strategy + 1) % len(self.cycle)

        count = 0
        for i in range(self.nstrategies):
            if npoints[i] > 0:
                new_points[count:count + npoints[i], :] = self.strategy_list[i].make_points(npts = npoints[i], xbest = xbest, sigma = sigma, front = front, subset = subset, projection = projection)

                count += npoints[i]
                for j in range(self.nstrategies):
                    if j != i:
                        self.strategy_list[j].proposed_points = self.strategy_list[i].proposed_points

        return new_points



class CandidateSRBF(object):
    def __init__(self, mo_problem, ncands=None, weights=None):
        self.mo_problem = mo_problem
        self.fhat = None
        self.xrange = self.mo_problem.ub - self.mo_problem.lb

        self.dtol = 1e-3 * math.sqrt(mo_problem.dim)
        self.precision = 1e-8 / math.sqrt(mo_problem.dim)

        if weights is None:
            self.weights = [0.3, 0.5, 0.8, 0.95]
        else:
            self.weights = weights

        self.proposed_points = None
        self.dmerit = None
        self.xcand = None
        self.fhvals = None
        self.next_weight = 0
        if ncands is None:
            self.ncands = min([5000, 100 * mo_problem.dim])
        else:
            self.ncands = ncands
        self.budget = None

    def init(self, start_sample, fhat, budget):
        self.proposed_points = start_sample
        self.fhat = fhat
        self.budget = budget

    def remove_point(self, x):
        index = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[index, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, index, axis=0)
            return True
        return False

    def __generate_cand__(self, scalefactors, xbest, subset):
        self.xcand = np.ones((self.ncands, self.mo_problem.dim)) * xbest
        if np.random.rand() <= 0.65:
            for i in subset:
                lb, ub = self.mo_problem.lb[i], self.mo_problem.ub[i]
                sigma = scalefactors[i]
                self.xcand[:, i] = stats.truncnorm.rvs((lb - xbest[i]) / sigma, (ub - xbest[i]) / sigma, loc=xbest[i], scale=sigma, size=self.ncands)
        else:
            for i in subset:
                lb, ub = self.mo_problem.lb[i], self.mo_problem.ub[i]
                sigma = scalefactors[i]
                self.xcand[:, i] = stats.norm.rvs(loc=xbest[i], scale=sigma, size=self.ncands)
                self.xcand[:, i] = np.minimum(ub, np.maximum(lb, self.xcand[:, i]))

    def make_points(self, npts, xbest, sigma, front, subset=None, projection=None):
        if subset is None:
            subset = np.arange(0, self.mo_problem.dim)
        scalefactors = sigma * self.xrange

        index = np.intersect1d(self.mo_problem.int_var, subset)
        if len(index) > 0:
            scalefactors[index] = np.maximum(scalefactors[index], 1.0)

        # -----------------------------------------------------
        # Candidate Generation
        self.__generate_cand__(scalefactors, xbest, subset)
        if projection is not None:
            self.xcand = projection(self.xcand)

        '''
        minimum_dis = INF
        for x in self.fhat[0].X:
            for y in self.fhat[0].X:
                if minimum_dis >= scp.distance.euclidean(x, y) and not (x == y).all():
                    minimum_dis = scp.distance.euclidean(x, y)
        print('minimum_dis = {}'.format(minimum_dis))
        '''

        fhvals = np.zeros((self.ncands, self.mo_problem.nobj))
        for i, fhat in enumerate(self.fhat):
            fvals = fhat.predict(self.xcand)
            fvals = fvals.flatten()
            fhvals[:, i] = fvals
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Candidate ND
        (ndf_index, df_index) = ND_front(np.transpose(fhvals))
        self.xcand_nd = self.xcand[ndf_index, :]
        self.fhvals_nd = fhvals[ndf_index, :]
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Candidate Selection
        if random.uniform(0, 1) <= 0.35:
            distance = scp.distance.cdist(self.xcand_nd, self.proposed_points)
            self.dmerit = np.amin(np.asmatrix(distance), axis=1)
            index = np.argmax(self.dmerit)
            xnew = self.xcand_nd[index, :]
        else:
            # Use a random solution from ND candidates
            (M, l) = self.xcand_nd.shape
            # index = random.randint(0, M - 1)
            # xnew = self.xcand_nd[index, :]

            # Use hypervolume contribution to selet the next best
            temp_all = np.vstack((self.fhvals_nd, front))
            minpt = np.zeros(self.mo_problem.nobj)
            maxpt = np.zeros(self.mo_problem.nobj)
            for i in range(self.mo_problem.nobj):
                minpt[i] = np.min(temp_all[:, i])
                maxpt[i] = np.max(temp_all[:, i])
            normalized_front = np.asarray(normalize_objectives(front, minpt, maxpt))
            (N, l) = normalized_front.shape
            normalized_fhvals_nd = np.asarray(normalize_objectives(self.fhvals_nd.tolist(), minpt, maxpt))

            hv = HyperVolume(1.1 * np.ones(self.mo_problem.nobj))
            base_hv = hv.compute(normalized_front)
            hv_vals = np.zeros(M)
            for i in range(M):
                nondominated = list(range(N))
                dominated = []
                fvals = np.vstack((normalized_front, normalized_fhvals_nd[i, :]))
                (nondominated, dominated) = ND_add(np.transpose(fvals), nondominated, dominated)
                if dominated and dominated[0] == N:
                    hv_vals[i] = 0
                else:
                    new_hv = hv.compute(fvals)
                    hv_vals[i] = new_hv - base_hv
            index = np.argmax(hv_vals)
            xnew = self.xcand_nd[index, :]

        '''
        Uniqueness = False
        while not Uniqueness:
            Uniqueness = True
            for point in self.proposed_points:
                if scp.distance.euclidean(point, xnew) <= self.precision:
                    Uniqueness = False
                    break
            if not Uniqueness:
                print('Found')
                scalefactors = sigma * self.xrange
                position = random.choice(list(range(self.mo_problem.dim)))
                lb, ub = self.mo_problem.lb[position], self.mo_problem.ub[position]
                temp_sigma = np.maximum(scalefactors[position], 1.0)
                xnew[position] = stats.truncnorm.rvs((lb - xnew[position]) / temp_sigma, (ub - xnew[position]) / temp_sigma, loc = xnew[position], scale = temp_sigma, size = 1)
        '''

        self.proposed_points = np.vstack((self.proposed_points, np.asmatrix(xnew)))
        # -----------------------------------------------------
        return xnew



class CandidateDYCORS(CandidateSRBF):
    def __init__(self, mo_problem, ncands=None, weights=None):
        CandidateSRBF.__init__(self, mo_problem, ncands=ncands, weights=weights)
        self.minprob = np.min([1.0, 1.0 / self.mo_problem.dim])
        self.n0 = None

        if mo_problem.dim <= 1:
            raise ValueError("Cannot use DYCORS on a 1d problem")

        def probfun(numevals, budget):
            if budget < 2:
                return 0
            return min([20.0 / mo_problem.dim, 1.0]) * (1.0 - (np.log(numevals + 1.0) / np.log(budget)))

        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        # ddsprob = 1.0
        # print(f"ddsprob = {ddsprob}")

        nlen = len(subset)

        if nlen == 1:
            ar = np.ones((self.ncands, 1))
        else:
            ar = (np.random.rand(self.ncands, nlen) < ddsprob)
            index = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[index, np.random.randint(0, nlen - 1, size=len(index))] = 1

        self.xcand = np.ones((self.ncands, self.mo_problem.dim)) * xbest
        if np.random.rand() <= 0.0:
            # print('CandidateDYCORS_Truncnorm')
            for i in range(nlen):
                lower, upper = self.mo_problem.lb[subset[i]], self.mo_problem.ub[subset[i]]
                sigma = scalefactors[subset[i]]
                index = np.where(ar[:, i] == 1)[0]
                # start_time = time.time()
                self.xcand[index, subset[i]] = stats.truncnorm.rvs((lower - xbest[subset[i]]) / sigma, (upper - xbest[subset[i]]) / sigma, loc=xbest[subset[i]], scale=sigma, size=len(index))
                # end_time = time.time()
                # print('Generate Time = {}'.format(end_time - start_time))
        else:
            # print('CandidateDYCORS_Norm')
            for i in range(nlen):
                lower, upper = self.mo_problem.lb[subset[i]], self.mo_problem.ub[subset[i]]
                sigma = scalefactors[subset[i]]
                index = np.where(ar[:, i] == 1)[0]
                self.xcand[index, subset[i]] = stats.norm.rvs(loc=xbest[subset[i]], scale=sigma, size=len(index))
                self.xcand[:, subset[i]] = np.minimum(upper, np.maximum(lower, self.xcand[:, subset[i]]))

    def make_points(self, npts, xbest, sigma, front, subset=None, projection=None):
        return CandidateSRBF.make_points(self, npts, xbest, sigma, front, subset, projection)



class CandidateDYUNIF(CandidateDYCORS):
    def __init__(self, mo_problem, ncands = None, weights = None):
        CandidateDYCORS.__init__(self, mo_problem, ncands = ncands, weights = weights)

    def init(self, start_sample, fhat, budget):
        CandidateDYCORS.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateDYCORS.remove_point(self, x)

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        if nlen == 1:
            ar = np.ones((self.ncands, 1))
        else:
            ar = (np.random.rand(self.ncands, nlen) < ddsprob)
            index = np.where(np.sum(ar, axis = 1) == 0)[0]
            ar[index, np.random.randint(0, nlen - 1, size = len(index))] = 1

        self.xcand = np.ones((self.ncands, self.mo_problem.dim)) * xbest

        # print('CandidateDYUNIF')
        for i in range(nlen):
            index = np.where(ar[:, i] == 1)[0]
            self.xcand[index, subset[i]] = np.random.uniform(self.mo_problem.lb[subset[i]], self.mo_problem.ub[subset[i]], size = len(index))

    def make_points(self, npts, xbest, sigma, front, subset = None, projection = None):
        return CandidateDYCORS.make_points(self, npts, xbest, sigma, front, subset, projection)



if __name__ == '__main__':
    mo_problem = ZDT1()
    sampling = CandidateSRBF(mo_problem = mo_problem)

    print(type(np.arange(0, mo_problem.dim)))
    print(3 * np.asarray([1, 2, 3]))
    print(np.asarray([[1, 2, 3], [4, 5, 6]]).flatten())

    ar = (np.random.rand(6, 4) < 0.65)
    print(ar)
    index = np.where(np.sum(ar, axis = 1) == 0)[0]
    print(index)
    ar[index, np.random.randint(0, 4 - 1, size = len(index))] = 1
    print(ar)

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(np.atleast_2d(a))
    b = np.array([1, 2, 3])
    print(np.atleast_2d(b))
    print(np.asmatrix(b))
    print(np.vstack((a, b)))