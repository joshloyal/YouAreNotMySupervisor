import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.manifold as manifold

from bokeh_plots import scatter_with_hover
from bokeh_server import bokeh_server


sns.set_context('poster')
sns.set_color_codes()
plot_kwargs = {'alpha': 0.25, 's': 50, 'linewidth': 0}
color_palette = sns.color_palette('deep', 8)


algorithm_class_dict = {
    'mds': manifold.MDS,
    'tsne': manifold.TSNE
}


algorithm_kwargs_dict = {
    'mds': dict(n_components=2, max_iter=100, n_init=1, random_state=0),
    'tsne': dict(n_components=2, init='pca', random_state=0),
}


def plot_2d(data, labels=None, probabilities=None, algorithm='tsne', algorithm_kwargs=None):
    if data.shape[1] > 2:
        algorithm_class = algorithm_class_dict[algorithm]
        if algorithm_kwargs:
            algorithm = algorithm_class(**algorithm_kwargs)
        else:
            algorithm = algorithm_class(**algorithm_kwargs_dict[algorithm])
        Y = algorithm.fit_transform(data)
    else:
        Y = data


    color_palette = sns.color_palette('deep', len(np.unique(labels)))
    if labels is not None:
        cluster_colors = [color_palette[x] if x >= 0 else
                          (0.5, 0.5, 0.5) for
                          x in labels]

        if probabilities is not None and np.isfinite(probabilities):
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, probabilities)]
        else:
            cluster_member_colors = cluster_colors
    else:
        cluster_member_colors = 'b'

    plt.scatter(Y[:, 0], Y[:, 1], c=cluster_member_colors, **plot_kwargs)
    frame = plt.gca()
    frame.get_xaxis().set_visible(False)
    frame.get_yaxis().set_visible(False)
    plt.show()


def bokeh_plot_2d(data, labels=None, probabilities=None, algorithm='tsne', algorithm_kwargs=None, untransformed_data=None):
    if data.shape[1] > 2:
        algorithm_class = algorithm_class_dict[algorithm]
        if algorithm_kwargs:
            algorithm = algorithm_class(**algorithm_kwargs)
        else:
            algorithm = algorithm_class(**algorithm_kwargs_dict[algorithm])
        Y = algorithm.fit_transform(data)
    else:
        Y = data


    color_palette = sns.color_palette('deep', len(np.unique(labels)))
    if labels is not None:
        cluster_colors = [color_palette[x] if x >= 0 else
                          (0.5, 0.5, 0.5) for
                          x in labels]

        if probabilities is not None and np.all(np.isfinite(probabilities)):
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, probabilities)]
        else:
            cluster_member_colors = cluster_colors

        cluster_member_colors = [mpl.colors.rgb2hex(rgb) for rgb in cluster_member_colors]
    else:
        cluster_member_colors = 'b'


    if untransformed_data is not None:
        original_columns = untransformed_data.columns.tolist()
        df = untransformed_data.copy()
        df['proj1'] = Y[:, 0]
        df['proj2'] = Y[:, 1]
    else:
        original_columns = []
        data_dict = {}
        for column in xrange(data.shape[1]):
            colname = 'x%i' % column
            original_columns.append(colname)
            data_dict[colname] = data[:, column]
        data_dict.update({'proj1': Y[:, 0], 'proj2': Y[:, 1]})
        df = pd.DataFrame(data_dict)

    with bokeh_server(name='comp') as server:
        q = scatter_with_hover(df, 'proj1', 'proj2', cols=original_columns, color=cluster_member_colors, alpha=0.5, size=10)
        server.show(q)
