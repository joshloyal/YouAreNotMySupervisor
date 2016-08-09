from hdbscan import HDBSCAN
from sklearn.cluster import MiniBatchKMeans, KMeans
from mysuper.datasets import fetch_kdd99

from plot import bokeh_plot_2d

data = fetch_kdd99(only_numeric=True)[:1000]
original = fetch_kdd99(original_dataframe=True, only_numeric=True)[:1000]


hdb = HDBSCAN(min_cluster_size=10, metric='l2').fit(data)
bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_, untransformed_data=original)

zero_data = original[ hdb.labels_ == 0 ]

#kmeans = MiniBatchKMeans(n_clusters=10, max_iter=10, random_state=1234).fit(data)
#bokeh_plot_2d(data, labels=kmeans.labels_, algorithm='tsne',  untransformed_data=original)
