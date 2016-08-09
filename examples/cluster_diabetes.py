from hdbscan import HDBSCAN
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import MiniBatchKMeans, KMeans

from mysuper.datasets import fetch_10kdiabetes
from plot import bokeh_plot_2d, plot_parallel_coordinates

data = fetch_10kdiabetes(only_numerics=True)[:1000]
#original = fetch_10kdiabetes(original_dataframe=True)[:1000]
#
#cluster = HDBSCAN(min_cluster_size=5, metric='l2').fit(data)
#bokeh_plot_2d(data.values, labels=cluster.labels_, probabilities=cluster.probabilities_, algorithm='tsne')
#
#
cluster = MiniBatchKMeans(n_clusters=5, n_init=20, max_iter=100, random_state=1234).fit(data.values)
bokeh_plot_2d(data.values, labels=cluster.labels_, algorithm='tsne')
plot_parallel_coordinates(data, labels=cluster.labels_, show_average=True)
