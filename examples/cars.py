from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from mysuper.datasets import fetch_cars

from plot import bokeh_plot_2d, plot_parallel_coordinates, project_data

data = fetch_cars()

#cluster = HDBSCAN(min_cluster_size=5, metric='l2').fit(data.values)
#bokeh_plot_2d(data, labels=hdb.labels_, probabilities=hdb.probabilities_)

cluster = KMeans(n_clusters=3, max_iter=100, random_state=1234).fit(data.values)
#bokeh_plot_2d(data, labels=cluster.labels_)
#plot_parallel_coordinates(data, labels=cluster.labels_)
Y = project_data(data.values, algorithm='tsne')
data['name'] = cluster.labels_
data['name'] = data['name'].apply(lambda x: 'group_{}'.format(x))
data['group'] = cluster.labels_
data['y1'] = Y[:, 0]
data['y2'] = Y[:, 1]
data.to_csv('test.csv', index_label='index')
