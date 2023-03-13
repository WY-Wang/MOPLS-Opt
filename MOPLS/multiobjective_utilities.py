import numpy as np
import scipy.spatial as scp
import numpy.matlib as npmat
from scipy.special import comb
import itertools

def domination(Fa, Fb, M):
    flag = False
    for i in range(0, M):
        if Fa[i] > Fb[i]:
            flag = False
            break
        elif Fa[i] < Fb[i]:
            flag = True
    return flag

def ND_front(F):
    (M, l) = F.shape
    df_index = []
    ndf_index = [int(0)]

    for i in range(1, l):
        (ndf_index, df_index) = ND_add(F[:, 0:i + 1], ndf_index, df_index)
    return (ndf_index, df_index)

def ND_add(F, ndf_index, df_index):
    (M, l) = F.shape
    l = int(l - 1)
    ndf_index.append(l)
    ndf_count = len(ndf_index)
    j = 1
    while j < ndf_count:
        if domination(F[:, l], F[:, ndf_index[j - 1]], M):
            df_index.append(ndf_index[j - 1])
            ndf_index.remove(ndf_index[j - 1])
            ndf_count -= 1
        elif domination(F[:, ndf_index[j - 1]], F[:, l], M):
            df_index.append(l)
            ndf_index.remove(l)
            ndf_count -= 1
            break
        else:
            j += 1
    return (ndf_index, df_index)

def epsilon_ND_front(F, epsilon):
    (M, l) = F.shape
    df_index = []
    box_df_index = []
    ndf_index = [int(0)]

    for i in range(1, l):
        (ndf_index, df_index, box_df_index) = epsilon_ND_add(F[:, 0:i + 1], ndf_index, df_index, box_df_index, epsilon)
    return (ndf_index, df_index, box_df_index)

def epsilon_ND_add(F, ndf_index, df_index, box_df_index, epsilon):
    (M, l) = F.shape
    l = int(l - 1)
    ndf_index.append(l)
    ndf_count = len(ndf_index)
    j = 1

    F_box = np.transpose(transform_into_boxes(np.transpose(F), epsilon))
    while j < ndf_count:
        if domination(F_box[:, l], F_box[:, ndf_index[j - 1]], M):
            df_index.append(ndf_index[j - 1])
            ndf_index.remove(ndf_index[j - 1])
            ndf_count -= 1
        elif domination(F_box[:, ndf_index[j - 1]], F_box[:, l], M):
            df_index.append(l)
            ndf_index.remove(l)
            ndf_count -= 1
            break
        elif np.array_equal(F_box[:, l], F_box[:, ndf_index[j - 1]]):
            norm1 = np.linalg.norm((F[:, l] - F_box[:, l]) / epsilon)
            norm2 = np.linalg.norm((F[:, ndf_index[j - 1]] - F_box[:, ndf_index[j - 1]]) / epsilon)
            if norm1 < norm2:
                box_df_index.append(ndf_index[j - 1])
                ndf_index.remove(ndf_index[j - 1])
                ndf_count -= 1
            else:
                box_df_index.append(l)
                ndf_index.remove(l)
                ndf_count -= 1
                break
        else:
            j += 1

    return (ndf_index, df_index, box_df_index)

def transform_into_boxes(F, epsilon):
    if not isinstance(epsilon, np.ndarray):
        raise ValueError("Numpy array of epsilon required")
    F_box = np.multiply(np.floor(F / epsilon), epsilon)
    return F_box

def normalize_objectives(fvals, minpt = None, maxpt = None):
    nobj = len(fvals[0])
    if maxpt is None:
        maxpt = [max([rec[i] for rec in fvals]) for i in range(nobj)]
    if minpt is None:
        minpt = [min([rec[i] for rec in fvals]) for i in range(nobj)]
    normalized_fvals = []
    for item in fvals:
        normalized_fvals.append([(item[i] - minpt[i]) / (maxpt[i] - minpt[i]) if maxpt[i] - minpt[i] > 0 else 0 for i in range(nobj)])
    return normalized_fvals

def radius_rule(record, center_pts, d_thresh):
    flag = True
    if center_pts == []:
        flag = True
    else:
        x_center = np.asarray([rec.x for rec in center_pts])
        sigmas = [rec.sigma for rec in center_pts]
        ncenters = len(center_pts)
        x = np.asarray(record.x)
        for i in range(ncenters):
            distance = scp.distance.euclidean(x, x_center[i, :])
            if distance < sigmas[i] * d_thresh / np.sqrt(len(x)):
                flag = False
                break
    return flag

def uniform_points(exp_npoints, nobj):
    H1 = 1
    while comb(H1 + nobj, nobj - 1) <= exp_npoints:
        H1 += 1
    W = np.subtract(np.array(list(itertools.combinations(list(range(H1 + nobj - 1)), nobj - 1))),
                    npmat.repmat(np.arange(nobj - 1), int(comb(H1 + nobj - 1, nobj - 1)), 1))
    W = np.subtract(np.hstack((W, np.zeros((len(W), 1)) + H1)), np.hstack((np.zeros((len(W), 1)), W))) / float(H1)

    if H1 < nobj:
        H2 = 0
        while comb(H1 + nobj - 1, nobj - 1) + comb(H2 + nobj, nobj - 1) <= exp_npoints:
            H2 += 1
        if H2 > 0:
            W2 = np.subtract(np.array(list(itertools.combinations(list(range(H2 + nobj - 1)), nobj - 1))),
                            npmat.repmat(np.arange(nobj - 1), int(comb(H2 + nobj - 1, nobj - 1)), 1))
            W2 = np.subtract(np.hstack((W2, np.zeros((len(W2), 1)) + H2)),
                             np.hstack((np.zeros((len(W2), 1)), W2))) / float(H2)
            W = np.vstack((W, W2 / 2.0 + 1.0 / (2.0 * nobj)))

    W = np.maximum(W, 1e-6)
    return W