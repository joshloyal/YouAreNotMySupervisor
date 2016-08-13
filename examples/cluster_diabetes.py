from hdbscan import HDBSCAN
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

from mysuper.datasets import fetch_10kdiabetes
from mysuper.unsupervised_grid_search import UnsupervisedCV, make_cluster_coherence_scorer
from plot import bokeh_plot_2d, plot_parallel_coordinates, prep_for_d3


n_samples = 10000

data = fetch_10kdiabetes()
if n_samples < data.shape[0]:
    data = data[:n_samples]

pca_X = RandomizedPCA(n_components=500, random_state=1234).fit_transform(data.values)
#cluster = HDBSCAN(min_cluster_size=2, metric='l2').fit(pca_X)
#bokeh_plot_2d(data.values, labels=cluster.labels_, probabilities=cluster.probabilities_, algorithm='tsne')

cluster = MiniBatchKMeans(n_init=20, max_iter=100, random_state=1234)
grid_search = GridSearchCV(
        cluster,
        param_grid={'n_clusters': [2, 3, 4]},
        cv=UnsupervisedCV(n_samples=int(data.values.shape[0])),
        scoring=make_cluster_coherence_scorer(metrics.silhouette_score),
        n_jobs=4)
grid_search.fit(pca_X)

#bokeh_plot_2d(data.values, labels=cluster.labels_, algorithm='tsne')
#plot_parallel_coordinates(data, labels=cluster.labels_, show_average=True)
#prep_for_d3(data, cluster, 'projected_diabetes.csv')
