from hdbscan import HDBSCAN
from mysuper.datasets import fetch_hdbscan_demo

from plot import plot_2d, bokeh_plot_2d

data = fetch_hdbscan_demo()

hdb = HDBSCAN(min_cluster_size=15, metric='l2').fit(data)
#bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_)
