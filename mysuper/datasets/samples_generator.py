import numpy as np


def mixed_data(n_samples=100):
    rng = np.random.RandomState(1234)

    n_rows = n_samples // 3
    remainder = n_samples % 3
    nominal = np.concatenate([
        rng.normal(0, 1, n_rows),
        rng.normal(1, 1, n_rows),
        rng.normal(2, 1, n_rows + remainder)])


    binary = np.concatenate([
        rng.binomial(1, .2, n_rows),
        rng.binomial(1, .5, n_rows),
        rng.binomial(1, .8, n_rows + remainder)])

    ordinal = np.concatenate([
        rng.binomial(5, .2, n_rows),
        rng.binomial(5, .5, n_rows),
        rng.binomial(5, .8, n_rows + remainder)])

    return np.c_[(nominal, binary, ordinal)]
