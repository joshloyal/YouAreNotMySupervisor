from hdbscan import HDBSCAN
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import MiniBatchKMeans, KMeans

from mysuper.datasets import fetch_10kdiabetes
from plot import bokeh_plot_2d

data = fetch_10kdiabetes()[:1000]
original = fetch_10kdiabetes(original_dataframe=True)[:1000]

#hdb = HDBSCAN(min_cluster_size=10, metric='cosine').fit(data)
#bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_, algorithm='tsne', untransformed_data=original)


kmeans = MiniBatchKMeans(n_clusters=5, max_iter=10, random_state=1234).fit(data)
bokeh_plot_2d(data, labels=kmeans.labels_, algorithm='tsne',  untransformed_data=original)
