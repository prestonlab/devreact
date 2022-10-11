"""Plotting manuscript figures."""

import os
import contextlib
import warnings
from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from devreact import model


def set_style(style_path=None):
    """Set default plot style."""
    if style_path is None:
        style_path = resource_filename('devreact', 'data/figures.mplstyle')
    plt.style.use(style_path)


def savefig(fig, fname, **kwargs):
    """Save figure while suppressing font subsetting errors."""
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            fig.savefig(fname, **kwargs)


def plot_age_param(
    trace, var_name, ages, age_ticks=None, absmax=None, color='C0', log=False, ax=None
):
    """Plot a parameter that varies with age with confidence bands."""
    if ax is None:
        ax = plt.gca()
    param = model.age_parameters(trace, [var_name])
    stats = model.age_stats(trace, var_name, ages)
    if log:
        stats['lower'] = np.exp(stats['lower'])
        stats['upper'] = np.exp(stats['upper'])
        stats['mean'] = np.exp(stats['mean'])
    sns.scatterplot(data=param, x='age', y='Value', color=color, ax=ax)
    ax.fill_between(
        stats['age'], stats['lower'], stats['upper'], color=color, alpha=0.1
    )
    ax.plot(stats['age'], stats['mean'], '-', color=color)
    ax.set(xlabel='Age (years)', ylabel=var_name)
    if age_ticks is not None:
        ax.set(xticks=age_ticks, xlim=(age_ticks[0], age_ticks[-1]))
    if absmax is not None:
        ax.axhline(0, 0, 1, linestyle='--', color='k')
        ax.set(ylim=(-absmax, absmax))
    return ax


def plot_age_params(trace, var_names):
    """Plot parameters that vary with age."""
    params = trace.posterior.mean(['chain', 'draw'])[var_names].to_dataframe()
    params['age'] = trace.constant_data.age.values
    res = pd.melt(
        params.reset_index(),
        id_vars=['subject', 'age'],
        value_vars=var_names,
        value_name='Value',
        var_name='Parameter',
    )
    g = sns.relplot(data=res, x='age', y='Value', col='Parameter')
    g.set(xlabel='Age (years)')
    g.set_titles(template='{col_name}')
    return g


def plot_signal_coef(trace, coefs, param_name, signal_names):
    """Plot coefficients relating signals to a parameter."""
    fig, ax = plt.subplots(1, len(coefs), figsize=(6 * len(coefs), 4))
    for i, coef in enumerate(coefs):
        var_names = [f'{param_name}_{s}_{coef}' for s in signal_names]
        az.plot_forest(trace, var_names=var_names, ax=ax[i], combined=True)
        if i > 0:
            ax[i].set_yticklabels('')
        else:
            ax[i].set_yticklabels(signal_names[::-1], fontsize='large')
        ax[i].set_xlabel(coef)
        ax[i].set_title('')
        abs_max = np.max(np.abs(ax[i].get_xlim()))
        ax[i].set(xlim=(-abs_max, abs_max))
        ax[i].axvline(0, *ax[i].get_ylim(), color='k')
    return fig, ax


def plot_predictive(
    predictive,
    group='posterior',
    row='trial_type',
    col='accuracy',
    trial_type=None,
    max_time=None,
    n_sample=50,
):
    """Plot prior or posterior predictive responses."""
    # get the relevant samples
    if group == 'prior':
        pps = predictive.prior_predictive
    elif group == 'posterior':
        pps = predictive.posterior_predictive
    else:
        raise ValueError(f'Invalid group: {group}')
    samples = pps.stack({'sample': ['chain', 'draw']})
    m = samples.dims['sample']

    factors = {'row': row, 'col': col}
    names = {}
    values = {}
    labels = {}
    for position, factor in factors.items():
        if factor == 'trial_type':
            n = np.array(['direct', 'inference'])
            v = n[predictive.constant_data.trial_type_index.values]
        elif factor == 'accuracy':
            n = np.array(['correct', 'incorrect'])
            v = n[1 - predictive.constant_data.x.values[:, 0].astype(int)]
        elif factor == 'age':
            n = np.array(['Age 7-8', 'Age 9-10', 'Age 11-12', 'Age 18-21'])
            sind = predictive.constant_data.subject_index.values
            v = n[predictive.constant_data.age_bin_index.values[sind]]
        else:
            raise ValueError(f'Invalid factor: {factor}.')
        names[position] = n
        values[position] = v
        labels[position] = [s.capitalize() for s in n]

    nr = len(np.unique(values['row']))
    nc = len(np.unique(values['col']))
    fig, ax = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=(3 * nc, 3 * nr))
    hist_kwargs = {
        'element': 'poly',
        'fill': False,
        'binrange': (0, max_time),
        'binwidth': 0.25,
        'stat': 'density',
    }
    pps_kwargs = {'linewidth': 0.5, 'alpha': 0.5, 'color': 'C0'}
    obs_kwargs = {'linewidth': 1, 'color': 'k'}

    # plot a subset of the predictive samples
    trial_types = np.array(['direct', 'inference'])
    tt = trial_types[predictive.constant_data.trial_type_index.values]
    ind = np.random.choice(m, n_sample)
    for s in range(n_sample):
        sample = samples.isel(sample=ind[s])
        times = sample.sel(component='response_time').response.values
        for i, R in enumerate(names['row']):
            for j, C in enumerate(names['col']):
                match = (values['row'] == R) & (values['col'] == C)
                if trial_type is not None:
                    match &= tt == trial_type
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.histplot(times[match], ax=ax[i, j], **hist_kwargs, **pps_kwargs)

    # plot the observed data
    times = predictive.observed_data.sel(component='response_time').response.values
    for i, R in enumerate(names['row']):
        for j, C in enumerate(names['col']):
            match = (values['row'] == R) & (values['col'] == C)
            if trial_type is not None:
                match &= tt == trial_type
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.histplot(times[match], ax=ax[i, j], **hist_kwargs, **obs_kwargs)

    for c in range(nc):
        ax[0, c].set(title=labels['col'][c], xlim=[0, max_time])
        ax[nr - 1, c].set(xlabel='Response time')
    for r in range(nr):
        ax[r, 0].set(ylabel=f'{labels["row"][r]} density')
    return fig, ax


def plot_predictive_acc(predictive, group='posterior'):
    """Plot predictive accuracy by subject."""
    if group == 'prior':
        pps = predictive.prior_predictive
    elif group == 'posterior':
        pps = predictive.posterior_predictive
    else:
        raise ValueError(f'Invalid group: {group}')
    pps = model.set_trial_coords(predictive.constant_data, pps)
    obs = model.set_trial_coords(predictive.constant_data, predictive.observed_data)
    p = (
        pps.sel(component='response')
        .mean(['chain', 'draw'])
        .to_dataframe()
        .groupby(['subject', 'trial_type'])['response']
        .mean()
    )
    o = (
        obs.sel(component='response')
        .to_dataframe()
        .groupby(['subject', 'trial_type'])['response']
        .mean()
    )
    accuracy = pd.concat([o, p], axis=1, keys=['Observed', 'Predictive']).reset_index()
    accuracy['trial_type'] = accuracy['trial_type'].str.capitalize()
    accuracy['age'] = np.repeat(
        predictive.constant_data.age.values, accuracy['trial_type'].nunique()
    )
    ticks = np.linspace(0, 1, 5)
    g = sns.relplot(
        data=accuracy,
        x='Predictive',
        y='Observed',
        hue='age',
        palette='crest',
        alpha=0.6,
        col='trial_type',
        clip_on=False,
        height=3.5,
        aspect=.8,
    )
    g.set(xlim=[0, 1], ylim=[0, 1], xticks=ticks, yticks=ticks)
    g.set_titles(template='{col_name}')
    return g


def plot_predictive_rt(predictive, group='posterior', max_time=None):
    """Plot predictive response time by subject."""
    obs = model.observed_dataframe(predictive)
    ppc = model.predictive_dataframe(predictive, group)

    groups = ['subject', 'trial_type', 'response_label']
    rt = pd.concat(
        [
            obs.groupby(groups)['response_time'].mean(),
            ppc.groupby(groups)['response_time'].mean(),
        ], axis=1, keys=['Observed', 'Predictive']
    ).reset_index()

    rt['trial_type'] = rt['trial_type'].str.capitalize()
    rt['age'] = np.repeat(
        predictive.constant_data.age.values,
        rt['trial_type'].nunique() * rt['response_label'].nunique(),
    )
    g = sns.relplot(
        data=rt,
        x='Predictive',
        y='Observed',
        hue='age',
        palette='crest',
        alpha=0.6,
        row='trial_type',
        col='response_label',
        clip_on=False,
        height=3.5,
        aspect=.8,
    )
    g.set_titles(template='{row_name} {col_name}')
    if max_time is not None:
        g.set(xlim=[0, max_time], ylim=[0, max_time])
    return g
