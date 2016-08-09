from hdbscan import HDBSCAN
import sklearn.datasets as datasets

from plot import bokeh_plot_2d, plot_parallel_coordinates

digits = datasets.load_digits()
data = digits.data
hdb = HDBSCAN(min_cluster_size=15).fit(data)

bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_, algorithm='pca')
#plot_parallel_coordinates(data, labels=hdb.labels_, n_components=10)
