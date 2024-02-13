"""Model memory retrieval decision responses."""

import warnings
import math
import numpy as np
import pandas as pd
import xarray as xr
import pytensor
import pytensor.tensor as pt
import pymc as pm
import arviz as az


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
    """Calculate probability density function for 3AFC using pytensor."""
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
    p = pt.switch(pt.eq(i, 1), c1, 2 * c2)
    return p


def pdf_separate(response_data, n, s, τ, A, b, v1, v2, v3, v4):
    """PDF for a separate-process model."""
    i = response_data[:, 0]
    t = response_data[:, 1] - τ

    # PDF for each accumulator
    v1a = pt.switch(pt.eq(n, 1), v1, v2)
    v2a = pt.switch(pt.eq(n, 1), v3, v4)
    p1 = tpdf(t, A, b, v1a, s)
    p2 = tpdf(t, A, b, v2a, s)

    # probability of having not hit threshold by now
    n1 = 1 - tcdf(t, A, b, v1a, s)
    n2 = 1 - tcdf(t, A, b, v2a, s)

    # conditional probability of each accumulator hitting threshold now
    c1 = p1 * n2 * n2
    c2 = p2 * n1 * n2

    # calculate probability of this response and rt
    p = pt.switch(pt.eq(i, 1), c1, 2 * c2)
    return p


def pdf_dual(response_data, n, s, τ, A, b, v1, v2, r, v3, v4):
    """PDF for a dual-process model."""
    i = response_data[:, 0]
    t = response_data[:, 1] - τ

    # PDF for each accumulator
    v2a = v2 * r ** (n - 1)
    v3a = pt.switch(pt.eq(n, 1), v3, v4)
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
    p = pt.switch(pt.eq(i, 1), c1 + c2, 2 * c3)
    return p


def function_pdf_single():
    """Generate a pytensor function to evaluate the PDF."""
    # time and response vary by trial
    response = pt.dmatrix('r')
    n = pt.ivector('n')

    # parameters are fixed over trial
    params = [
        pt.dscalar('s'),
        pt.dscalar('τ'),
        pt.dscalar('A'),
        pt.dscalar('b'),
        pt.dscalar('v1'),
        pt.dscalar('v2'),
    ]

    p = pdf_single(response, n, *params)
    f = pytensor.function([response, n, *params], p, on_unused_input='ignore')
    return f


def function_pdf_separate():
    """Generate a pytensor function to evaluate the separate-model PDF."""
    # time and response vary by trial
    response = pt.dmatrix('r')
    n = pt.ivector('n')

    # parameters are fixed over trial
    params = [
        pt.dscalar('s'),
        pt.dscalar('τ'),
        pt.dscalar('A'),
        pt.dscalar('b'),
        pt.dscalar('v1'),
        pt.dscalar('v2'),
        pt.dscalar('v3'),
        pt.dscalar('v4'),
    ]

    p = pdf_separate(response, n, *params)
    f = pytensor.function([response, n, *params], p)
    return f


def function_pdf_dual():
    """Generate a pytensor function to evaluate the dual-model PDF."""
    # time and response vary by trial
    response = pt.dmatrix('r')
    n = pt.ivector('n')

    # parameters are fixed over trial
    params = [
        pt.dscalar('s'),
        pt.dscalar('τ'),
        pt.dscalar('A'),
        pt.dscalar('b'),
        pt.dscalar('v1'),
        pt.dscalar('v2'),
        pt.dscalar('r'),
        pt.dscalar('v3'),
        pt.dscalar('v4'),
    ]

    p = pdf_dual(response, n, *params)
    f = pytensor.function([response, n, *params], p)
    return f


def logp_single(response_data, n, *params):
    """Calculate log probability using pytensor."""
    p = pdf_single(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(pm.math.clip(p, 10e-10, np.Inf)))
    return ll


def logp_separate(response_data, n, *params):
    """Calculate log probability using pytensor."""
    p = pdf_separate(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(pm.math.clip(p, 10e-10, np.Inf)))
    return ll


def logp_dual(response_data, n, *params):
    """Calculate log probability using pytensor."""
    p = pdf_dual(response_data, n, *params)
    ll = pm.math.sum(pm.math.log(pm.math.clip(p, 10e-10, np.Inf)))
    return ll


def age_var(name, age, coef_mu, coef_sigma, beta, log=False):
    """
    Generate a parameter that varies with age.

    Parameters
    ----------
    name : str
        Name of the parameter to vary with age.

    age : pymc.ConstantData
        Mean-centered age in years.

    coef_mu : numpy.ArrayLike
        Mean of each group-level age coefficient.

    coef_sigma : numpy.ArrayLike
        Standard deviation of each group-level age coefficient.

    beta : float
        Spread of the half-Cauchy distribution for subject-level sampling.

    log : bool
        If true, sampled parameters will be treated as log values and
        exponentiated in the output variable.

    Returns
    -------
    param : pymc.Deterministic
        The age-varying parameter for each subject, determined using a
        non-centered parameterization.
    """
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


def age_signal_var(names, age, coef_mu, coef_sigma, beta, signal_data, signals):
    """
    Generate a parameter whose relationship to a signal varies with age.

    Parameters
    ----------
    names : list of str
        Names of parameters that depend on the signals.

    age : pymc.ConstantData
        Mean-centered age in years.

    coef_mu : numpy.ArrayLike
        Mean of each group-level age coefficient determining how the slope
        that determines how the parameter varies with changes in the signal
        changes with age.

    coef_sigma : numpy.ArrayLike
        Standard deviation of each group-level age coefficient.

    beta : float
        Spread of the half-Cauchy distribution for subject-level sampling.

    signal_data : dict of (str: numpy.ArrayLike)
        Mean-centered values of each signal on each trial. Set missing data to
        zero so that the signal will not influence the parameter on those
        trials.

    signals : list of str
        Signals to include as predictors.

    Returns
    -------
    x : dict of (str: pymc.ConstantData)
        Signal data cast to pymc.ConstantData.

    coef : dict of (str: dict of (str: pymc.Deterministic))
        Subject-level slope for each signal, parameter combination.
    """
    x = {}
    coef = {}
    for signal in signals:
        s = signal_data[signal]
        data = pt.as_tensor(s)
        data_unobs = pm.Normal(f"{signal}_unobs", 0, 1, shape=(np.isnan(s).sum(),))
        data = pt.set_subtensor(data[np.isnan(s)], data_unobs)

        x[signal] = pm.Deterministic(signal, data, dims=['trial'])
        coef[signal] = {}
        for name in names:
            coef[signal][name] = age_var(
                f'{name}_{signal}', age, coef_mu, coef_sigma, beta
            )
    return x, coef


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


def random_dual(n, s, τ, A, b, v1, v2, r, v3, v4, rng, size=None):
    """Randomly sample based on a dual-process model."""
    if size is None:
        size = (n.shape[0], 2)

    τ = np.clip(broadcast2d(τ), 10e-10, 10e10)
    A = np.clip(broadcast2d(A), 10e-10, 10e10)
    b = np.clip(broadcast2d(b), 10e-10, 10e10)

    v2a = v2 * r ** (n - 1)
    v3a = np.where(n == 1, v3, v4)
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


def get_data_coords(data):
    """Get data coordinates for modeling."""
    subject, subject_index = np.unique(data['subject'].to_numpy(), return_inverse=True)

    trial_type = ['direct', 'indirect']
    trial_type_index = data['trial_type'].map({'XY': 0, 'BC': 0, 'AC': 1}).to_numpy()

    trial = np.arange(data.shape[0])

    coords = {
        'component': ['response', 'response_time'],
        'subject': subject,
        'trial_type': trial_type,
        'trial': trial,
    }
    return coords, subject_index, trial_type_index


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


def observed_means_dataframe(data):
    """Get dataframe of mean response time by condition and accuracy."""
    stats = observed_dataframe(data)
    m = (
        stats.groupby(['subject', 'trial_type', 'response_label']).mean()
        .reset_index()
        .drop(columns=['response'])
        .rename(columns={'response_label': 'accuracy'})
    )
    m.columns.name = None
    return m


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


def predictive_means_dataframe(
    data, group='posterior', max_time=None, min_responses=None
):
    """Get dataframe of mean response time by condition and accuracy."""
    if group == 'prior':
        pps = data.prior_predictive
    elif group == 'posterior':
        pps = data.posterior_predictive
    else:
        raise ValueError(f'Invalid group: {group}')
    pps = set_trial_coords(data.constant_data, pps)

    # create trial index
    df_index = pd.DataFrame(
        {
            'subject': pps.response.coords['subject'].values,
            'trial_type': pps.response.coords['trial_type'].values,
        }
    )
    df_index['subject_trial'] = df_index.groupby(['subject', 'trial_type']).cumcount()
    mi = pd.MultiIndex.from_frame(df_index)

    # reshape to get subject, trial type, trial, and sample coordinates
    shaped = (
        pps.response.assign_coords(trial=mi)
        .stack(sample=('chain', 'draw'))
        .unstack('trial')
    )

    # unpack coordinates
    response = shaped.sel(component='response')
    response_time = shaped.sel(component='response_time')

    # initialize data array with means
    acc = ['Incorrect', 'Correct']
    rtm = xr.DataArray(
        coords=[shaped.coords['subject'], shaped.coords['trial_type'], acc],
        dims=['subject', 'trial_type', 'accuracy'],
    )

    # calculate mean response time
    for subject in rtm.coords['subject'].values:
        for trial_type in rtm.coords['trial_type'].values:
            # samples for all trials in this condition
            sind = dict(subject=subject, trial_type=trial_type)
            r = response.loc[sind].values
            rt = response_time.loc[sind].values

            # remove responses over time limit
            if max_time is not None:
                # if the mean response time for a sample was late (rare), exclude
                exclude = np.nanmean(rt, 1) > max_time
                rt[exclude, :] = np.nan

            for a, accuracy in enumerate(rtm.coords['accuracy'].values):
                # remove responses for other accuracy bin
                temp = rt.copy()
                temp[r != a] = np.nan

                if min_responses is not None:
                    # exclude samples if less than minimal number of responses
                    n = np.sum(~np.isnan(temp), axis=1)
                    temp[n < min_responses, :] = np.nan

                # calculate mean over trials, then over samples
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    m = np.nanmean(np.nanmean(temp, axis=1))
                ind = dict(
                    subject=subject,
                    trial_type=trial_type,
                    accuracy=accuracy,
                )
                rtm.loc[ind] = m
    stats = rtm.to_dataframe(name='response_time').reset_index()
    return stats


def response_time_stats(
    predictive, group='posterior', max_time=None, min_responses=None
):
    """Observed and predictive response time statistics."""
    # mean response times for observed and predictive samples
    obs_mean = observed_means_dataframe(predictive)
    pps_mean = predictive_means_dataframe(
        predictive, group=group, max_time=max_time, min_responses=min_responses
    )

    # merge into one dataframe
    groups = ['subject', 'trial_type', 'accuracy']
    rt = (
        pd.merge(obs_mean, pps_mean, on=groups, how='outer')
        .rename(
            columns={'response_time_x': 'Observed', 'response_time_y': 'Predictive'}
        )
    )
    rt['trial_type'] = rt['trial_type'].str.capitalize()

    # add age column
    rt['age'] = np.repeat(
        predictive.constant_data.age.values,
        rt['trial_type'].nunique() * rt['accuracy'].nunique(),
    )
    return rt


def response_time_trial(predictive, group='posterior', max_time=None):
    """Observed and predictive response times for each trial."""
    if group == 'prior':
        pps = predictive.prior_predictive
    elif group == 'posterior':
        pps = predictive.posterior_predictive
    else:
        raise ValueError(f'Invalid group: {group}')
    pps = set_trial_coords(predictive.constant_data, pps)

    # observed trials
    obs = set_trial_coords(predictive.constant_data, predictive.observed_data)
    obs_mean = obs.sel(component='response_time').to_dataframe().reset_index()

    # mean over predictive samples for each trial
    pps = set_trial_coords(predictive.constant_data, pps)
    samples = pps.sel(component='response_time').stack(dict(sample=['chain', 'draw']))
    sample_mean = samples.mean('trial')

    # exclude samples whose mean over trials is past the response deadline
    m = samples.loc[dict(sample=sample_mean.response <= max_time)].mean('sample')
    pps_mean = m.to_dataframe().reset_index()

    # merge into one dataframe
    groups = ['subject', 'trial_type', 'component', 'trial']
    rt = (
        pd.merge(obs_mean, pps_mean, on=groups, how='outer')
        .rename(columns={'response_x': 'Observed', 'response_y': 'Predictive'})
    )
    rt['trial_type'] = rt['trial_type'].str.capitalize()

    # add age column
    subject_index = predictive.constant_data.subject_index.values
    rt['age'] = predictive.constant_data.age.values[subject_index]
    bin_names = np.array(['Age 7-8', 'Age 9-10', 'Age 11-12', 'Age 18-35'])
    age_bin_index = predictive.constant_data.age_bin_index.values[subject_index]
    rt['age_bin'] = bin_names[age_bin_index]
    rt['age_bin'] = rt['age_bin'].astype('category').cat.set_categories(
        bin_names, ordered=True
    )
    return rt


def age_parameters(trace, var_names):
    """Get mean posterior for parameters that vary with age."""
    params = trace.posterior[var_names].mean(['chain', 'draw']).to_dataframe()
    params['age'] = trace.constant_data.age.values
    results = pd.melt(
        params.reset_index(),
        id_vars=['subject', 'age'],
        value_vars=var_names,
        value_name='Value',
        var_name='Parameter',
    )
    return results


def age_stats(trace, var_name, ages):
    """Calculate parameter statistics as a function of age."""
    # stack all posterior samples together
    samples = trace.posterior.stack(dict(sample=['chain', 'draw']))

    # get age coefficients for this parameter
    coef = []
    n = 0
    while True:
        coef_name = f'{var_name}_b{n}'
        if coef_name in samples:
            coef.append(samples[coef_name].values[:, np.newaxis])
            n += 1
        else:
            break

    # for each sample, calculate mean as a function of age
    a = ages - np.mean(trace.constant_data.age.values)
    if len(coef) == 2:
        μ = coef[0] + coef[1] * a
    elif len(coef) == 3:
        μ = coef[0] + coef[1] * a + coef[2] * a ** 2
    else:
        raise ValueError('Unsupported number of age coefficients.')

    # mean over samples
    mean = np.mean(μ, axis=0)

    # high-density intervals
    na = len(ages)
    lower = np.zeros(na)
    upper = np.zeros(na)
    for i in range(na):
        hdi = az.hdi(μ[:, i])
        lower[i] = hdi[0]
        upper[i] = hdi[1]

    # package stats as a function of age
    stats = pd.DataFrame({'age': ages, 'mean': mean, 'lower': lower, 'upper': upper})
    return stats


def parameter_table(trace, param_map):
    """Create a table of posterior parameter estimates."""
    # get statistics
    var_names = list(param_map.keys())
    var_labels = list(param_map.values())
    m = trace.posterior[var_names].mean(['chain', 'draw']).to_array().to_series()
    hdi = az.hdi(trace.posterior[var_names])
    lower = hdi.sel({'hdi': 'lower'})[var_names].to_array().to_series()
    upper = hdi.sel({'hdi': 'higher'})[var_names].to_array().to_series()

    # create table
    s_hdi = [f'[{l:.4f}, {u:.4f}]' for l, u in zip(lower, upper)]
    table = pd.DataFrame({'Mean': m.to_numpy(), '94\% HDI': s_hdi}, index=var_labels)
    table.index.name = 'Parameter'
    table = table.reset_index()
    return table
