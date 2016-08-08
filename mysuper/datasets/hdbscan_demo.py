import os

import numpy as np

root = os.path.abspath(os.path.dirname(__file__))
FILE_NAME = os.path.join(root, 'clusterable_data.npy')


def fetch_hdbscan_demo():
    data = np.load(FILE_NAME)
    return data
