"""Utilities for reading task data."""

import numpy as np
import pandas as pd


def read_remind(data_file):
    """Read raw Remind data."""
    raw = pd.read_csv(data_file)

    # label trials with no response
    response = raw['accuracy'].copy()
    response[raw['response_time'].isna()] = np.nan

    # define age bins
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
            'subject': raw['subject'],
            'age': raw['age'],
            'age_centered': raw['age'] - raw['age'].mean(),
            'age_bin': age_bin,
            'age_label': age_bin.cat.set_categories(age_labels, rename=True),
            'run': raw['trial_type'].map(run_map),
            'trial': raw['trial'],
            'triad': raw['tiad'],
            'trial_type': raw['trial_type'].map(trial_type_map),
            'category': raw['trial_type'].map(category_map),
            'response': response,
            'response_label': response.map(response_map),
            'response_time': raw['response_time'] / 1000,
        }
    )
    return data
