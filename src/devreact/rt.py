"""Reponse time modeling for the Garnet study."""

import numpy as np
import pandas as pd
import theano.tensor as tt
from psireact import lba


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


def pdf_3afc(t, i, n, A, b, v1, v2, s):
    """Probability distribution function for 3afc tests."""
    # probability that all accumulators are negative
    all_neg = (
        lba.normcdf(-v1 / s) * lba.normcdf(-v2 / s) * lba.normcdf(-v2 / s)
    )

    # PDF for each accumulator
    p1 = lba.tpdf(t, A, b, v1, s)
    p2 = lba.tpdf(t, A, b, v2, s)

    # probability of having not hit threshold by now
    n1 = 1 - lba.tcdf(t, A, b, v1, s)
    n2 = 1 - lba.tcdf(t, A, b, v2, s)

    # conditional probability of each accumulator hitting threshold now
    c1 = p1 * n2 * n2
    c2 = p2 * n1 * n2
    c3 = p2 * n1 * n2

    # calculate probability of this response and rt,
    # conditional on a valid response
    pdf = tt.switch(
        tt.eq(i, 1), c1 / (1 - all_neg), (c2 + c3) / (1 - all_neg)
    )
    pdf_cond = tt.switch(tt.gt(t, 0), pdf, 0)
    return pdf_cond


def rvs_3afc(n, A, b, v1, v2, s, tau, size=1):
    """Random generator for 3afc tests."""

    # finish times of accumulators
    rt, resp = lba.sample_response(A, b, [v1, v2, v2], s, tau, size)

    # accumulator 1 indicates correct response
    correct = np.zeros(size)
    correct[resp == 0] = 1
    return rt, correct
