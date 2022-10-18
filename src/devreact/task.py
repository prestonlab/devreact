"""Utilities for reading task data."""

import numpy as np
import pandas as pd


def read_remind(data_file, signals=None, signal_names=None):
    """Read raw Remind data."""
    raw = pd.read_table(data_file)

    # label trials with no response
    response = raw['acc'].copy()
    response[raw['rt'].isna()] = np.nan

    # define age bins
    if 'precise_age_days' in raw:
        raw['age'] = raw['precise_age_days'] / 365
    age_bins = [6, 8, 10, 12, 35]
    age_labels = ['7-8', '9-10', '11-12', '18-35']
    age_bin = pd.cut(raw['age'], age_bins)

    # relabel and place into dataframe
    run_map = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
    trial_type_map = {1: 'BC', 2: 'BC', 3: 'XY', 4: 'AC', 5: 'AC'}
    category_map = {1: 'face', 2: 'scene', 3: None, 4: None, 5: None}
    response_map = {0: 'incorrect', 1: 'correct'}
    data = pd.DataFrame(
        {
            'subject': raw['id'],
            'age': raw['age'],
            'age_centered': raw['age'] - raw['age'].mean(),
            'age_bin': age_bin,
            'age_label': age_bin.cat.set_categories(age_labels, rename=True),
            'run': raw['run'].map(run_map),
            'trial': raw['trial'],
            'triad': raw['triad'],
            'trial_type': raw['test_type'].map(trial_type_map),
            'category': raw['test_type'].map(category_map),
            'response': response,
            'response_label': response.map(response_map),
            'response_time': raw['rt'] / 1000,
        }
    )
    if signals is not None:
        names = signal_names if signal_names is not None else signals
        for signal, name in zip(signals, names):
            data[name] = raw[signal]
    return data


def read_kidrep(csv_file):
    """Read response time data."""
    data = pd.read_csv(csv_file)
    data['response_time'] /= 1000
    data.trial_type = data.trial_type.map({1: 'AB', 2: 'BC', 3: 'XY', 4: 'AC'})
    data = data.astype({'trial_type': 'category'})
    data.trial_type.cat.reorder_categories(
        ['AB', 'XY', 'BC', 'AC'], ordered=True, inplace=True
    )
    return data
