"""Model memory retrieval decision responses."""

import math
import numpy as np
import pandas as pd
import aesara
import aesara.tensor as at
import pymc as pm


def normpdf(x):
    return (1 / pm.math.sqrt(2 * math.pi)) * pm.math.exp(-(x ** 2) / 2)


def normcdf(x):
    return (1 / 2) * (1 + pm.math.erf(x / pm.math.sqrt(2)))


def tpdf(t, A, b, v, s):
    """Probability distribution function over time."""
    g = (b - A - t * v) / (t * s)
    h = (b - t * v) / (t * s)
    f = (-v * normcdf(g) + s * normpdf(g) + v * normcdf(h) - s * normpdf(h)) / A
    return f


def tcdf(t, A, b, v, s):
    """Cumulative distribution function over time."""
    e1 = ((b - A - t * v) / A) * normcdf((b - A - t * v) / (t * s))
    e2 = ((b - t * v) / A) * normcdf((b - t * v) / (t * s))
    e3 = ((t * s) / A) * normpdf((b - A - t * v) / (t * s))
    e4 = ((t * s) / A) * normpdf((b - t * v) / (t * s))
    F = 1 + e1 - e2 + e3 - e4
    return F


def pdf_single(response_data, n, s, τ, A, b, v1, v2):
    """Calculate probability density function for 3AFC using Aesara."""
    i = response_data[:, 0]
    t = response_data[:, 1] - τ

    # PDF for each accumulator
    p1 = tpdf(t, A, b, v1, s)
    p2 = tpdf(t, A, b, v2, s)

    # probability of having not hit threshold by now
    n1 = 1 - tcdf(t, A, b, v1, s)
    n2 = 1 - tcdf(t, A, b, v2, s)

    # conditional probability of each accumulator hitting threshold now
    c1 = p1 * n2 * n2
    c2 = p2 * n1 * n2

    # calculate probability of this response and rt
    p = at.switch(at.eq(i, 1), c1, 2 * c2)
    return p


def pdf_dual(response_data, n, s, τ, A, b, v1, v2, r, v3):
    """PDF for a dual-process model."""
    i = response_data[:, 0]
    t = response_data[:, 1] - τ

    # PDF for each accumulator
    v2a = v2 * r ** (n - 1)
    p1 = tpdf(t, A, b, v1, s)
    p2 = tpdf(t, A, b, v2a, s)
    p3 = tpdf(t, A, b, v3, s)

    # probability of having not hit threshold by now
    n1 = 1 - tcdf(t, A, b, v1, s)
    n2 = 1 - tcdf(t, A, b, v2a, s)
    n3 = 1 - tcdf(t, A, b, v3, s)

    # conditional probability of each accumulator hitting threshold now
    c1 = p1 * n2 * n3 * n3
    c2 = p2 * n1 * n3 * n3
    c3 = p3 * n1 * n2 * n3

    # calculate probability of this response and rt
    p = at.switch(at.eq(i, 1), c1 + c2, 2 * c3)
    return p


def function_pdf_single():
    """Generate an Aesara function to evaluate the PDF."""
    # time and response vary by trial
    response = at.dmatrix('r')
    n = at.ivector('n')

    # parameters are fixed over trial
    params = [
        at.dscalar('s'),
        at.dscalar('τ'),
        at.dscalar('A'),
        at.dscalar('b'),
        at.dscalar('v1'),
        at.dscalar('v2'),
    ]

    p = pdf_single(response, n, *params)
    f = aesara.function([response, n, *params], p, on_unused_input='ignore')
    return f


def function_pdf_dual():
    """Generate an Aesara function to evaluate the dual-model PDF."""
    # time and response vary by trial
    response = at.dmatrix('r')
    n = at.ivector('n')

    # parameters are fixed over trial
    params = [
        at.dscalar('s'),
        at.dscalar('τ'),
        at.dscalar('A'),
        at.dscalar('b'),
        at.dscalar('v1'),
        at.dscalar('v2'),
        at.dscalar('r'),
        at.dscalar('v3'),
    ]

    p = pdf_dual(response, n, *params)
    f = aesara.function([response, n, *params], p)
    return f


def logp_single(response_data, n, *params):
    """Calculate log probability using Aesara."""
    p = pdf_single(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(p))
    return ll


def logp_dual(response_data, n, *params):
    """Calculate log probability using Aesara."""
    p = pdf_dual(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(p))
    return ll


def drift_rates(v, s, nt, rng):
    """Generate random drift rates."""
    # sample drift rates with constraint that, on each trial,
    # at least one accumulator must be positive
    nv = len(v)
    d = np.zeros((nt, nv))
    valid = False
    isvalid = np.zeros(nt, dtype=bool)
    while not valid:
        # generate random accumulator drift rates
        c = np.zeros((nt, nv))
        for i in range(nv):
            c[:, i] = rng.normal(loc=v[i], scale=s, size=nt)
        valid_c = c[np.any(c > 0, axis=1), :]

        # replace invalid trials
        n = np.count_nonzero(~isvalid)
        m = valid_c.shape[0]
        k = n if n <= m else m
        replace_ind = np.where(~isvalid)[0]
        d[replace_ind[:k], :] = valid_c[:k, :]

        # check if all valid
        isvalid = np.any(d > 0, axis=1)
        valid = np.all(isvalid)
    d[d <= 0] = np.nan
    return d


def random_single(n, s, τ, A, b, v1, v2, rng, size=None):
    """Randomly sample correct and incorrect response times."""
    if size is None:
        size = (1, 2)

    # sample start point and drift on each trial
    k = rng.uniform(0, A, size=(size[0], 3))
    d = drift_rates([v1, v2, v2], s, size[0], rng)
    t = τ + ((b - k) / d)

    # score responses (0 is correct, 1 or 2 is incorrect)
    x = np.zeros((size[0], 2))
    winner = np.nanargmin(t, axis=1)
    x[winner == 0, 0] = 1
    x[winner != 0, 0] = 0

    # time first accumulator finished
    x[:, 1] = np.nanmin(t, axis=1)
    return x


def random_dual(n, s, τ, A, b, v1, v2, r, v3, rng, size=None):
    """Randomly sample based on a dual-process model."""
    if size is None:
        size = (n.shape[0], 2)

    τ = np.asarray(τ)[:, np.newaxis]
    A = np.asarray(A)[:, np.newaxis]
    b = np.asarray(b)[:, np.newaxis]

    v2a = v2 * r ** (n - 1)
    k = rng.uniform(0, A, size=(size[0], 4))
    d = drift_rates([v1, v2a, v3, v3], s, size[0], rng)
    t = τ + ((b - k) / d)
    x = np.zeros((size[0], 2))
    winner = np.nanargmin(t, axis=1)
    x[winner <= 1, 0] = 1
    x[winner >= 2, 0] = 0
    x[:, 1] = np.nanmin(t, axis=1)
    return x


def response_dataframe(mat, trial_type=None):
    """Convert a response matrix to DataFrame format."""
    data = pd.DataFrame({'Response': mat[:, 0], 'Response time': mat[:, 1]})
    data['Response'] = data['Response'].map({0: 'Incorrect', 1: 'Correct'})
    if trial_type is not None:
        data['Trial type'] = trial_type
        data['Trial type'] = data['Trial type'].map({1: 'Direct', 2: 'Indirect'})
    return data
