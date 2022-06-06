"""Plotting manuscript figures."""

from pkg_resources import resource_filename
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    pps_kwargs = {'linewidth': 0.5, 'alpha': 0.5, 'color': 'C0', 'warn_singular': False}
    obs_kwargs = {'linewidth': 1, 'color': 'k'}

    # plot a subset of the predictive samples
    ind = np.random.choice(m, n_sample)
    for s in range(n_sample):
        sample = samples.isel(sample=ind[s])
        responses = sample.sel(component='response').response.values
        times = sample.sel(component='response_time').response.values
        for i, N in enumerate([1, 2]):
            for j, R in enumerate([1, 0]):
                sns.kdeplot(
                    times[(n == N) & (responses == R)], ax=ax[i, j], **pps_kwargs
                )

    # plot the observed data
    responses = predictive.observed_data.sel(component='response').response.values
    times = predictive.observed_data.sel(component='response_time').response.values
    for i, N in enumerate([1, 2]):
        for j, R in enumerate([1, 0]):
            sns.kdeplot(times[(n == N) & (responses == R)], ax=ax[i, j], **obs_kwargs)
    ax[0, 0].set(title='Correct', xlabel='Response time')
    ax[0, 1].set(title='Incorrect', xlabel='Response time')
    ax[0, 0].set(xlim=[0, max_time], ylabel='Direct density')
    ax[0, 1].set(xlim=[0, max_time])
    ax[1, 0].set(xlabel='Response time', ylabel='Indirect density')
    ax[1, 1].set(xlabel='Response time')
