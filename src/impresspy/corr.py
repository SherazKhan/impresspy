from __future__ import division
from scipy.spatial.distance import pdist, squareform, correlation
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import random
import copy
from scipy.stats import norm



def distcorr(Xval, Yval, pval=True, nruns=500):
    """ Compute the distance correlation function, returning the p-value.
    Based on Satra/distcorr.py (gist aa3d19a12b74e9ab7941)
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    (0.76267624241686671, 0.268)
    """
    X = np.atleast_1d(Xval)
    Y = np.atleast_1d(Yval)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        for i in range(nruns):
            Y_r = copy.copy(Yval)
            np.random.shuffle(Y_r)
            if distcorr(Xval, Y_r, pval=False) >= dcor:
                greater += 1
        return (dcor, greater/float(nruns))
    else:
        return dcor


def mult_comp_corr(X, y, n_perm=1000, method = 'spearman'):

    if X.ndim == 1:
        if method == 'spearman':
            (r, p) = spearmanr(X, y)
        elif method == 'distance':
            (r, p) = distcorr(X, y)
        else:
            (r, p) = pearsonr(X, y)
        return r, p

    else:
        if method == 'spearman':
            r, p = zip(*[spearmanr(x, y) for x in X.T])
        elif method == 'distance':
            r, p = zip(*[distcorr(x, y) for x in X.T])
        else:
            r, p = zip(*[pearsonr(x, y) for x in X.T])

    if n_perm > 0:
        progress = 0
        r_p = np.zeros(n_perm)
        for index in range(n_perm):
            y_r = copy.copy(y)
            random.shuffle(y_r)
            r_p[index] = np.abs(mult_comp_corr(X, y_r, n_perm=0)[0]).max()
            progress += 1.
            print('Running {:.4f} completed'.format(100 * progress / n_perm))

        p = np.zeros(X.shape[1])
        for index in range(X.shape[1]):
            p[index] = (np.sum(np.int32(r_p >= np.abs(r[index])))+1.)/(n_perm+1.)

        return r, p
    else:
        return r, p

def compare_corr(x,y,groups,n_perm=1000):
    if n_perm==0:
        corr_diff, p =compare_correlation_coefficients(pearsonr(x[groups == 0], y[groups == 0])[0],
                                                   pearsonr(x[groups == 1], y[groups == 1])[0],
                                                   np.sum(groups == 0),np.sum(groups == 1))
        return corr_diff, p
    else:
        corr_diff = compare_corr(x, y, groups, n_perm=0)[0]
        greater = 0.0
        for i in range(n_perm):
            groups_r = copy.copy(groups)
            random.shuffle(groups_r)
            if np.abs(compare_corr(x, y, groups_r,n_perm=0)[0]) >= np.abs(corr_diff):
                greater += 1.
        return corr_diff, (greater+1) / (n_perm+1)



def compare_correlation_coefficients(r1,r2,n1,n2):
    t_r1 = 0.5*np.log((1+r1)/(1-r1))
    t_r2 = 0.5*np.log((1+r2)/(1-r2))
    z = (t_r1-t_r2)/np.sqrt(1/(n1-3)+1/(n2-3))
    p = (1-norm.cdf(np.abs(z),0,1))*2
    return z, p




def mat_corr(mat, method = 'spearman', diagnol = False, n_perm=0):
    p = np.zeros(mat.shape)
    r = np.zeros(mat.shape)

    for ind1 in range(mat.shape[1]):
        for ind2 in range(mat.shape[1]):
            if method == 'spearman':
                r[ind1, ind2], p[ind1, ind2] = \
                    spearmanr(mat[:,ind1], mat[:,ind2])
            elif method == 'pearson':
                r[ind1, ind2], p[ind1, ind2] = \
                    pearsonr(mat[:,ind1], mat[:,ind2])
            elif method == 'kendal':
                r[ind1, ind2], p[ind1, ind2] = \
                    kendalltau(mat[:,ind1], mat[:,ind2])

    if not diagnol:
        np.fill_diagonal(p, 1)
        np.fill_diagonal(r, 0)

    if n_perm > 0:
        progress = 0
        r_p = np.zeros(n_perm)
        for index in range(n_perm):
            mat_r = copy.copy(mat)
            for ind in range(mat.shape[1]):
                mat_r[:, ind] = mat_r[np.random.permutation(mat.shape[0]), ind]
            temp = abs(mat_corr(mat_r, method=method, diagnol=False, n_perm=0)[0])
            r_p[index] = temp[np.triu_indices_from(temp, 1)].max()

            progress += 1.
            print('Running {:.4f} completed'.format(100 * progress / n_perm))

        p = np.zeros(r.shape)
        for index1 in range(r.shape[0]):
            for index2 in range(r.shape[1]):
                p[index1, index2] = \
                    (np.sum(np.int32(r_p >= np.abs(r[index1, index2])))+1.)/(n_perm+1.)

        return r, p
    else:
        return r, p
































































