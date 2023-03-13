import numpy as np
from copy import deepcopy
#from hv import HyperVolume
from pyhv import HyperVolume
from multiobjective_utilities import ND_add, normalize_objectives, radius_rule

INF = float('inf')


class Memory_Record():
    def __init__(self, x, fx, sigma, nfail=0, ntabu=0, rank=INF, fitness=INF):
        self.x = x
        self.fx = fx
        self.sigma_init = sigma
        self.sigma = sigma
        self.nfail = nfail
        self.ntabu = ntabu
        self.rank = rank
        self.fitness = fitness

        self.generated = 0
        self.noffsprings = 1
        self.offsprings = []
        self.fhat_pts = []

    def reset(self):
        self.nfail = 0
        self.ntabu = 0
        self.generated = 0
        self.sigma = self.sigma_init


class Memory_Archive():
    def __init__(self, size_max):
        self.contents = []
        self.size_max = size_max
        self.num_records = 0

    def add(self, record, cur_rank=None):
        if cur_rank is None:
            cur_rank = 1
        if self.contents:
            ranked = False
            while cur_rank <= len(self.contents):
                front = self.contents[cur_rank - 1]

                fvals = [rec.fx for rec in front]
                nrecords = len(fvals)
                nondominated = list(range(nrecords))
                dominated = []
                fvals.append(record.fx)
                fvals = np.asarray(fvals)
                (nondominated, dominated) = ND_add(np.transpose(fvals), nondominated, dominated)
                if dominated == []:
                    ranked = True
                    record.rank = cur_rank
                    front.append(record)
                    for item in front:
                        item.fitness = INF
                    self.num_records += 1
                    break
                elif dominated[0] == nrecords:
                    fvals = None
                else:
                    ranked = True
                    record.rank = cur_rank
                    front.append(record)
                    self.num_records += 1
                    dominated = sorted(dominated, reverse=True)
                    for i in dominated:
                        dominated_record = deepcopy(front[i])
                        front.remove(front[i])
                        self.num_records -= 1
                        self.add(dominated_record, cur_rank)
                    for item in front:
                        item.fitness = INF
                    break
                cur_rank += 1

            if ranked == False:
                record.rank = len(self.contents) + 1
                record.fitness = INF
                self.contents.append([record])
                self.num_records += 1

        else:
            self.contents.append([record])
            self.num_records += 1
            record.rank = 1
            record.fitness = INF

        if self.num_records > self.size_max:
            self.contents[-1].remove(self.contents[-1][-1])
            if self.contents[-1] == []:
                self.contents.remove(self.contents[-1])
            self.num_records -= 1

    def compute_hv_fitness(self, cur_rank):
        front = deepcopy(self.contents[cur_rank - 1])
        nrecords = len(front)
        if nrecords == 1:
            self.contents[cur_rank - 1][0].fitness = 1
        else:
            fvals = [rec.fx for rec in front]
            nobj = len(front[0].fx)
            normalized_fvals = normalize_objectives(fvals)
            hv = HyperVolume(1.1 * np.ones(nobj))
            base_hv = hv.compute(np.asarray(normalized_fvals))
            for i in range(nrecords):
                fvals_without = deepcopy(normalized_fvals)
                fvals_without.remove(fvals_without[i])
                new_hv = hv.compute(np.asarray(fvals_without))
                self.contents[cur_rank - 1][i].fitness = base_hv - new_hv

    def select_center_population(self, npts, d_thresh=1.0):
        center_pts = []
        count = 1
        nfronts = len(self.contents)
        cur_rank = 1
        flag_tabu = False

        while count <= npts:
            front = self.contents[cur_rank - 1]
            if front[0].fitness == INF:
                self.compute_hv_fitness(cur_rank)
            front.sort(key=lambda x: x.fitness, reverse=True)
            for rec in front:
                if rec.generated == 0:
                    if flag_tabu == True:
                        rec.reset()
                        center_pts.append(rec)
                        count += 1
                        if count > npts:
                            break
                    elif rec.ntabu == 0:
                        flag_radius = radius_rule(rec, center_pts, d_thresh)
                        if flag_radius == True:
                            center_pts.append(rec)
                            count += 1
                            if count > npts:
                                break
            cur_rank = int((cur_rank % nfronts) + 1)
            if cur_rank == 1:
                flag_tabu = True
            if cur_rank == 1 and flag_tabu:
                break

        return center_pts

