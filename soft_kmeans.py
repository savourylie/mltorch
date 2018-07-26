import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
from sklearn.metrics import normalized_mutual_info_score

from mpl_toolkits.mplot3d import Axes3D

NUM_CLUSTERS = 10
LABEL_COLOR_MAP = cm.rainbow(np.linspace(0, 1, NUM_CLUSTERS))

X, y = make_blobs(n_samples=50000, centers=NUM_CLUSTERS, n_features=2, cluster_std=1, center_box=(-20.0, 20.0))

print(X.shape)

class HardKmeans(object):
    def __init__(self, n_clusters=4, n_init=10, max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(self.random_state)


    def fit(self, X):
        inertia_cluster_by_round = []

        for it in range(self.n_init):
            print("Epoch {}:".format(it))
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

            init_center_indices = np.random.randint(0, high=self.n_rows, size=self.n_clusters)
            init_centers = X[init_center_indices]
            diff_arrays = [X - center for center in init_centers]

            inertia_array = np.array([sum([diff_arr[:, col] ** 2 for col in range(self.n_cols)]) for diff_arr in diff_arrays]).transpose()
            # print(inertia_array.shape)
            inertia_sum = inertia_array.sum()

            cluster = np.argmin(inertia_array, axis=1)
            # cluster_temp = np.zeros_like(cluster)
            counter = 0

            while True:
                centers = [X[np.where(cluster == c), :].mean() for c in range(self.n_clusters)]
                diff_arrays = [X - center for center in centers]
                inertia_array = np.array([sum([diff_arr[:, col] ** 2 for col in range(self.n_cols)]) for diff_arr in diff_arrays]).transpose()
                inertia_sum_temp = inertia_array.sum()
                cluster_temp = np.argmin(inertia_array, axis=1)

                if abs(inertia_sum - inertia_sum_temp) < self.tol:
                    break
                else:
                    cluster = cluster_temp
                    inertia_sum = inertia_sum_temp
                    counter += 1

                print("Iteration: {}".format(counter))

                if counter > self.max_iter:
                    break

            inertia_cluster_by_round.append((inertia_sum, cluster))

        inertia_list = [inertia for inertia, _ in inertia_cluster_by_round]
        inertia_max = np.max(inertia_list)
        inertia_min = np.min(inertia_list)
        inertia_mean = np.mean(inertia_list)
        inertia_std = np.std(inertia_list)

        cluster = sorted(inertia_cluster_by_round, key=lambda x: x[0])[0][1]
        print("Inertia max: {}, Inertia min: {}, Max/min ratio: {}".format(inertia_max, inertia_min, inertia_max/inertia_min))
        print("Inertia mean: {0}, Inertia STD: {1}".format(inertia_mean, inertia_std))
        print(cluster)
        print(normalized_mutual_info_score(y, cluster))

        label_color = [LABEL_COLOR_MAP[l] for l in cluster]

        # LABEL_COLOR_MAP = {0: 'r', 1: 'g'}
        # LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b', 3: 'k'}
        # label_color = [LABEL_COLOR_MAP[l] for l in cluster]

        # # plot2d


        plt.scatter(X[:, 0], X[:, 1], c=label_color)
        plt.show()

        # plot3d
        # Xs = X[:,0]
        # Ys = X[:,1]
        # Zs = X[:,2]
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.scatter(Xs, Ys, Zs, c=label_color)
        #
        # plt.show() # or:

kmeans = HardKmeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

