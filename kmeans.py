import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
from sklearn.metrics import normalized_mutual_info_score
import torch
import sys
import time

from mpl_toolkits.mplot3d import Axes3D

NUM_CLUSTERS = 5
LABEL_COLOR_MAP = cm.rainbow(np.linspace(0, 1, NUM_CLUSTERS))

X, y = make_blobs(n_samples=50000, centers=NUM_CLUSTERS, n_features=2, cluster_std=1, center_box=(-50.0, 50.0))

print(X.shape)

class KMeans(object):
    def __init__(self, n_clusters=4, n_init=10, max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X):
        start_time = time.time()
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
                centers = [X[np.where(cluster == c), :].reshape(-1, self.n_cols).mean(0) for c in range(self.n_clusters)]
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

        print("Clustering took {} sec.".format(time.time() - start_time))
        label_color = [LABEL_COLOR_MAP[l] for l in cluster]

        # LABEL_COLOR_MAP = {0: 'r', 1: 'g'}
        # LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b', 3: 'k'}
        # label_color = [LABEL_COLOR_MAP[l] for l in cluster]

        # # plot2d


        # plt.scatter(X[:, 0], X[:, 1], c=label_color)
        # plt.show()

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

    def transform(self, X):
        pass

class KMeansTorch(object):
    def __init__(self, n_clusters=4, n_init=10, max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        if self.random_state:
            torch.manual_seed(self.random_state)

    def fit(self, X):
        start_time = time.time()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        if torch.cuda.is_available():
            X = X.cuda()

        inertia_cluster_by_round = []

        for it in range(self.n_init):
            print("Epoch {}:".format(it))
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

            if torch.cuda.is_available():
                init_center_indices = torch.randint(0, self.n_rows, (self.n_clusters,)).type(torch.LongTensor).cuda()
            else:
                init_center_indices = torch.randint(0, self.n_rows, (self.n_clusters, )).type(torch.LongTensor)

            init_centers = X[init_center_indices]
            diff_arrays = [X - center for center in init_centers]

            if torch.cuda.is_available():
                inertia_array = torch.stack(
                    [sum([diff_arr[:, col] ** 2 for col in range(self.n_cols)]) for diff_arr in diff_arrays]).transpose(
                    0, 1).cuda()
            else:
                inertia_array = torch.stack([sum([diff_arr[:, col] ** 2 for col in range(self.n_cols)]) for diff_arr in diff_arrays]).transpose(0, 1)

            inertia_sum = inertia_array.sum()
            _, cluster = inertia_array.min(1)
            counter = 0

            while True:
                centers = [X[(cluster == c).nonzero(), :].mean(0) for c in range(self.n_clusters)]
                diff_arrays = [X - center for center in centers]

                if torch.cuda.is_available():
                    inertia_array = torch.stack(
                        [sum([diff_arr[:, col] ** 2 for col in range(self.n_cols)]) for diff_arr in
                         diff_arrays]).transpose(0, 1).cuda()
                else:
                    inertia_array = torch.stack([sum([diff_arr[:, col] ** 2 for col in range(self.n_cols)]) for diff_arr in diff_arrays]).transpose(0, 1)

                inertia_sum_temp = inertia_array.sum()
                _, cluster_temp = inertia_array.min(1)

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
        print("Clustering took {} sec.".format(time.time() - start_time))

        label_color = [LABEL_COLOR_MAP[l] for l in cluster]

        # LABEL_COLOR_MAP = {0: 'r', 1: 'g'}
        # LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b', 3: 'k'}
        # label_color = [LABEL_COLOR_MAP[l] for l in cluster]

        # # plot2d


        # plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=label_color)
        # plt.show()

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

    def transform(self, X):
        pass

kmeans = KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

print("WTF1")

kmeans_torch = KMeansTorch(n_clusters=NUM_CLUSTERS)
kmeans_torch.fit(X)

print("WTF2")


