from hdbscan import HDBSCAN

from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans

from mysuper.datasets import mixed_data
from plot import plot_2d, bokeh_plot_2d


data = mixed_data(n_samples=100)


hdb = HDBSCAN(min_cluster_size=5, metric='l2').fit(data)
bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_, algorithm='mds')

#kmeans = KMeans(n_clusters=3, random_state=1234).fit(data)
#bokeh_plot_2d(data, labels=kmeans.labels_,  algorithm='mds')
