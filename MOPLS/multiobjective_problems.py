import time
import math
from multiobjective_utilities import uniform_points
from pySOT.optimization_problems import OptimizationProblem
from scipy.stats import pareto
import numpy as np


class DTLZ1(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 4
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ1"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum(
            [math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim - k:]]))
        f = [0.5 * (1.0 + g)] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= solution[j]
            if i > 0:
                f[i] *= 1 - solution[self.nobj-i-1]
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            points /= 2.0
        return J


class DTLZ2(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ2"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution[self.nobj - i - 1])
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            temp = math.sqrt(sum([point ** 2 for point in points]))
            points /= float(temp)
        return J


class DTLZ3(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ3"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum(
            [math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim - k:]]))
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution[self.nobj - i - 1])
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            temp = math.sqrt(sum([point ** 2 for point in points]))
            points /= float(temp)
        return J


class DTLZ4(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ4"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        alpha = 100.0
        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * math.pow(solution[j], alpha))
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * math.pow(solution[self.nobj - i - 1], alpha))
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            temp = math.sqrt(sum([point ** 2 for point in points]))
            points /= float(temp)
        return J


class DTLZ5(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ5"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim-k:]])
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(1, self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[j]))

            if i > 0:
                if self.nobj-i-1 != 0:
                    f[i] *= math.sin(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[self.nobj-i-1]))
                else:
                    f[i] *= math.sin(0.5 * math.pi * solution[0])

            if self.nobj - i - 1 != 0:
                f[i] *= math.cos(0.5 * math.pi * solution[0])

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        P = [list(np.linspace(0, 1, exp_npoints)), list(np.linspace(1, 0, exp_npoints))]
        P = np.asarray(P).transpose()

        P = [[row[0] / np.sqrt(row[0] ** 2 + row[1] ** 2), row[1] / np.sqrt(row[0] ** 2 + row[1] ** 2)] for row in P]
        newP = []
        for row in P:
            newrow = []
            for _ in range(self.nobj - 1):
                newrow.append(row[0])
            newrow.append(row[1])
            newP.append(newrow)

        temp = list(range(self.nobj - 1, -1, -1))
        print(temp)
        temp[0] -= 1
        P = np.divide(newP, np.power(np.sqrt(2.0), npmat.repmat(temp, exp_npoints, 1)))
        return P


class DTLZ6(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ6"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x, 0.1) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(1, self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[j]))

            if i > 0:
                if self.nobj - i - 1 != 0:
                    f[i] *= math.sin(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[self.nobj - i - 1]))
                else:
                    f[i] *= math.sin(0.5 * math.pi * solution[0])

            if self.nobj - i - 1 != 0:
                f[i] *= math.cos(0.5 * math.pi * solution[0])

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        P = [list(np.linspace(0, 1, exp_npoints)), list(np.linspace(1, 0, exp_npoints))]
        P = np.asarray(P).transpose()

        P = [[row[0] / np.sqrt(row[0] ** 2 + row[1] ** 2), row[1] / np.sqrt(row[0] ** 2 + row[1] ** 2)] for row in P]
        newP = []
        for row in P:
            newrow = []
            for _ in range(self.nobj - 1):
                newrow.append(row[0])
            newrow.append(row[1])
            newP.append(newrow)

        temp = list(range(self.nobj - 1, -1, -1))
        temp[0] -= 1
        P = np.divide(newP, np.power(np.sqrt(2.0), npmat.repmat(temp, exp_npoints, 1)))
        return P


class DTLZ7(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None, delay = False, alpha = 102.0, xm = 1.0):
        if dim is None:
            dim = nobj + 19
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        # self.pf = self.paretofront(5000)

        self.delay = delay
        self.alpha = alpha
        self.xm = xm
        self.name = "DTLZ7"

    def eval(self, solution):
        if self.delay:
            delay = (float(np.random.pareto(self.alpha, size=1)) + 1) * self.xm
            # print("time delay = {}\n".format(delay))
            time.sleep(delay)

        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 1.0 + (sum([x for x in solution[self.dim-k:]])) * 9.0 / float(k)
        f = [1.0]*self.nobj

        for i in range(self.nobj):
            if i < self.nobj - 1:
                f[i] = solution[i]
            else:
                h = 0
                for j in range(self.nobj - 1):
                    h += f[j] / (1.0 + g) * (1.0 + np.sin(3.0 * np.pi * f[j]))
                h = self.nobj - h
                f[i] = (1.0 + g) * h

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])

        if self.nobj > 2:
            exp_npoints = int(np.ceil(exp_npoints ** (1.0 / (self.nobj - 1))) ** (self.nobj - 1))
            num = exp_npoints ** (1.0 / (self.nobj - 1))
            Gap = list(np.linspace(0, 1, int(num)))
            num = min([num, len(Gap)])
            exp_npoints = int(num ** (self.nobj - 1))

            label = [0] * (self.nobj - 1)
            X = []

            for i in range(exp_npoints):
                row = []
                for j in range(self.nobj - 1):
                    row.append(Gap[int(label[j])])
                X.append(row)

                label[self.nobj - 2] += 1
                for j in range(self.nobj - 2, 0, -1):
                    label[j - 1] += label[j] // num
                    label[j] = label[j] % num

            for i in range(len(X)):
                for j in range(len(X[0])):
                    if X[i][j] > median:
                        X[i][j] = (X[i][j] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
                    else:
                        X[i][j] = X[i][j] * (interval[1] - interval[0]) / median + interval[0]

            Xnew = [[2.0 * (self.nobj - sum([x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)) for x in row]))] for row in X]
            P = np.hstack((X, Xnew))
        else:
            X = list(np.linspace(0, 1, exp_npoints))
            for i in range(len(X)):
                if X[i] > median:
                    X[i] = (X[i] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
                else:
                    X[i] = X[i] * (interval[1] - interval[0]) / median + interval[0]
            Xnew = [2.0 * (self.nobj - x / 2.0 * (1 + np.sin(3 * np.pi * x))) for x in X]
            P = np.vstack((X, Xnew))
            P = P.transpose()

        return P





if __name__ == '__main__':
    mo_problem = DTLZ5(nobj = 6, dim = 10)
    print(mo_problem.eval(
        [0.812053, 0.754300, 0.683037, 0.074330, 0.890933, 0.172467, 0.985907, 0.637004, 0.304736, 0.621536]))
