"""Plotting manuscript figures."""

import warnings
from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from devreact import model


def set_style(style_path=None):
    """Set default plot style."""
    if style_path is None:
        style_path = resource_filename('devreact', 'data/figures.mplstyle')
    plt.style.use(style_path)


def plot_predictive(predictive, group='posterior', max_time=None, n_sample=50):
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

    # trial type
    n = predictive.constant_data.n.values

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
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
    ind = np.random.choice(m, n_sample)
    for s in range(n_sample):
        sample = samples.isel(sample=ind[s])
        responses = sample.sel(component='response').response.values
        times = sample.sel(component='response_time').response.values
        for i, N in enumerate([1, 2]):
            for j, R in enumerate([1, 0]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.histplot(
                        times[(n == N) & (responses == R)],
                        ax=ax[i, j],
                        **hist_kwargs,
                        **pps_kwargs,
                    )

    # plot the observed data
    responses = predictive.observed_data.sel(component='response').response.values
    times = predictive.observed_data.sel(component='response_time').response.values
    for i, N in enumerate([1, 2]):
        for j, R in enumerate([1, 0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.histplot(
                    times[(n == N) & (responses == R)],
                    ax=ax[i, j],
                    **hist_kwargs,
                    **obs_kwargs,
                )
    ax[0, 0].set(title='Correct', xlabel='Response time')
    ax[0, 1].set(title='Incorrect', xlabel='Response time')
    ax[0, 0].set(xlim=[0, max_time], ylabel='Direct density')
    ax[0, 1].set(xlim=[0, max_time])
    ax[1, 0].set(xlabel='Response time', ylabel='Indirect density')
    ax[1, 1].set(xlabel='Response time')


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
    ticks = np.linspace(0, 1, 5)
    g = sns.relplot(
        data=accuracy,
        x='Predictive',
        y='Observed',
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
    g = sns.relplot(
        data=rt,
        x='Predictive',
        y='Observed',
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
