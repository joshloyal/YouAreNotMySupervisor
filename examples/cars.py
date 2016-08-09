from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from mysuper.datasets import fetch_cars

from plot import bokeh_plot_2d, plot_parallel_coordinates

data = fetch_cars()
original = fetch_cars(no_processing=True)

#cluster = HDBSCAN(min_cluster_size=5, metric='l2').fit(data.values)
#bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_)

cluster = KMeans(n_clusters=3, max_iter=100, random_state=1234).fit(data.values)
#bokeh_plot_2d(data, labels=cluster.labels_)
plot_parallel_coordinates(data, labels=cluster.labels_)
