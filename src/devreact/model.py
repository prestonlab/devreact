"""Model memory retrieval decision responses."""

import math
import numpy as np
import pandas as pd
import aesara
import aesara.tensor as at
import pymc as pm


def interval_forward(x, a, b):
    """Transform a bounded variable to an unbounded scale."""
    return np.log(x - a) - np.log(b - x)


def interval_reverse(x, a, b):
    """Transform an unbounded variable back to a bounded scale."""
    s = 1 / (1 + np.exp(-x))
    return s * b + (1 - s) * a


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


def pdf_separate(response_data, n, s, τ, A, b, v1, v2, v3, v4):
    """PDF for a separate-process model."""
    i = response_data[:, 0]
    t = response_data[:, 1] - τ

    # PDF for each accumulator
    v1a = at.switch(at.eq(n, 1), v1, v2)
    v2a = at.switch(at.eq(n, 1), v3, v4)
    p1 = tpdf(t, A, b, v1a, s)
    p2 = tpdf(t, A, b, v2a, s)

    # probability of having not hit threshold by now
    n1 = 1 - tcdf(t, A, b, v1a, s)
    n2 = 1 - tcdf(t, A, b, v2a, s)

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
    v3a = v3 * r ** (n - 1)
    p1 = tpdf(t, A, b, v1, s)
    p2 = tpdf(t, A, b, v2a, s)
    p3 = tpdf(t, A, b, v3a, s)

    # probability of having not hit threshold by now
    n1 = 1 - tcdf(t, A, b, v1, s)
    n2 = 1 - tcdf(t, A, b, v2a, s)
    n3 = 1 - tcdf(t, A, b, v3a, s)

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


def function_pdf_separate():
    """Generate an Aesara function to evaluate the separate-model PDF."""
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
        at.dscalar('v3'),
        at.dscalar('v4'),
    ]

    p = pdf_separate(response, n, *params)
    f = aesara.function([response, n, *params], p)
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
    ll = pm.math.sum(pm.math.log(pm.math.clip(p, 10e-10, np.Inf)))
    return ll


def logp_separate(response_data, n, *params):
    """Calculate log probability using Aesara."""
    p = pdf_separate(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(pm.math.clip(p, 10e-10, np.Inf)))
    return ll


def logp_dual(response_data, n, *params):
    """Calculate log probability using Aesara."""
    p = pdf_dual(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(pm.math.clip(p, 10e-10, np.Inf)))
    return ll


def age_var(name, age, coef_mu, coef_sigma, beta, log=False):
    """Generate a parameter that varies with age."""
    # coefficients to determine mean over age
    age_b = []
    for i, (mu, sigma) in enumerate(zip(coef_mu, coef_sigma)):
        age_b.append(pm.Normal(f'{name}_b{i}', mu=mu, sigma=sigma))

    # determine mean as a function of age
    if len(age_b) == 2:
        subject_mu = pm.Deterministic(
            f'{name}_μ', age_b[0] + age_b[1] * age, dims=['subject']
        )
    elif len(age_b) == 3:
        subject_mu = pm.Deterministic(
            f'{name}_μ',
            age_b[0] + age_b[1] * age + age_b[2] * age ** 2,
            dims=['subject'],
        )
    else:
        raise ValueError('Unsupported number of age coefficients.')

    # determine subject parameters
    sigma = pm.HalfCauchy(f'{name}_σ', beta=beta)
    centered = pm.Normal(f'{name}_c', mu=0, sigma=1, dims=['subject'])
    if log:
        param = pm.Deterministic(
            name, pm.math.exp(subject_mu + sigma * centered), dims=['subject']
        )
    else:
        param = pm.Deterministic(name, subject_mu + sigma * centered, dims=['subject'])
    return param


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


def broadcast2d(x):
    """Broadcast a variable to a matrix."""
    x = np.asarray(x)
    if x.ndim == 0:
        x.shape = (1, 1)
    elif x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim > 2:
        raise ValueError('x must have 2 or fewer dimensions')
    return x


def random_single(n, s, τ, A, b, v1, v2, rng, size=None):
    """Randomly sample correct and incorrect response times."""
    if size is None:
        size = (n.shape[0], 2)

    τ = np.clip(broadcast2d(τ), 10e-10, 10e10)
    A = np.clip(broadcast2d(A), 10e-10, 10e10)
    b = np.clip(broadcast2d(b), 10e-10, 10e10)

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


def random_separate(n, s, τ, A, b, v1, v2, v3, v4, rng, size=None):
    """Randomly sample based on a separate-process model."""
    if size is None:
        size = (n.shape[0], 2)

    τ = np.clip(broadcast2d(τ), 10e-10, 10e10)
    A = np.clip(broadcast2d(A), 10e-10, 10e10)
    b = np.clip(broadcast2d(b), 10e-10, 10e10)

    v1a = np.where(n == 1, v1, v2)
    v2a = np.where(n == 1, v3, v4)
    k = rng.uniform(0, A, size=(size[0], 3))
    d = drift_rates([v1a, v2a, v2a], s, size[0], rng)
    t = τ + ((b - k) / d)
    x = np.zeros((size[0], 2))
    winner = np.nanargmin(t, axis=1)
    x[winner == 0, 0] = 1
    x[winner != 0, 0] = 0
    x[:, 1] = np.nanmin(t, axis=1)
    return x


def random_dual(n, s, τ, A, b, v1, v2, r, v3, rng, size=None):
    """Randomly sample based on a dual-process model."""
    if size is None:
        size = (n.shape[0], 2)

    τ = np.clip(broadcast2d(τ), 10e-10, 10e10)
    A = np.clip(broadcast2d(A), 10e-10, 10e10)
    b = np.clip(broadcast2d(b), 10e-10, 10e10)

    v2a = v2 * r ** (n - 1)
    v3a = v3 * r ** (n - 1)
    k = rng.uniform(0, A, size=(size[0], 4))
    d = drift_rates([v1, v2a, v3a, v3a], s, size[0], rng)
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


def set_trial_coords(constant_data, data):
    """Set coordinates based on trial labels."""
    c = constant_data
    trial_type = c.coords['trial_type'].data[c.trial_type_index.data]
    subject = c.coords['subject'].data[c.subject_index.data]
    data = data.assign_coords(
        subject=('trial', subject), trial_type=('trial', trial_type)
    )
    return data


def observed_dataframe(data):
    """Convert observed data to DataFrame format."""
    observed_data = set_trial_coords(data.constant_data, data.observed_data)
    df = pd.pivot(
        observed_data.response.to_dataframe().reset_index(),
        index=['trial', 'subject', 'trial_type'],
        columns='component',
        values='response',
    )
    df['response_label'] = df['response'].map({0: 'Incorrect', 1: 'Correct'})
    return df


def predictive_dataframe(data, group='posterior'):
    """Convert predictive data to DataFrame format."""
    if group == 'prior':
        pps = data.prior_predictive
    elif group == 'posterior':
        pps = data.posterior_predictive
    else:
        raise ValueError(f'Invalid group: {group}')
    pps = set_trial_coords(data.constant_data, pps)
    df = pd.pivot(
        pps.response.to_dataframe().reset_index(),
        index=['chain', 'draw', 'trial', 'subject', 'trial_type'],
        columns='component',
        values='response',
    )
    df['response_label'] = df['response'].map({0: 'Incorrect', 1: 'Correct'})
    return df
