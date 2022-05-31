"""Reponse time modeling for the Garnet study."""

import pandas as pd


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
