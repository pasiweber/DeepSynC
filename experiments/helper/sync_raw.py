# import os
# import math
import numpy as np
from copy import deepcopy
# from sklearn import datasets
# from sklearn.manifold import TSNE
# from dataset import get_dataLoader
from collections import deque

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
import imageio
from io import BytesIO


class SyncClus():
    def __init__(self, threshold=0.1, max_iter=100, tolerance=0.0001):
        self.threshold = threshold
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.labels_ = None

    def fit_predict(self, data, epsilon=0.1):
        old_rc = 0
        # imgs = []
        for i in range(self.max_iter):
            dist_fn = self.euclidean_distances
            delta, dist = dist_fn(data, data)
            neighbor = dist < self.threshold
            num_neighbor = neighbor.sum(0).unsqueeze(-1)
            index = neighbor.unsqueeze(-1).repeat(1, 1, data.shape[-1])
            delta[~index] = 0

            # if i%1 == 0: #(i < 50 and i%2==0) or (i>=50 and i%10==0)
            #     x = data.data.cpu().numpy()
            #     fig = plt.figure()
            #     plt.scatter(x[:, 0], x[:, 1], marker='o')
            #     plt.title('iter %d'%i)

            #     buffer_ = BytesIO()
            #     plt.savefig(buffer_, format='png')
            #     buffer_.seek(0)
            #     img = imageio.imread(buffer_)
            #     imgs.append(img)
            #     plt.close()

            data += torch.sin(delta).sum(0) / num_neighbor  #

            rc = self.get_rc(dist, neighbor)
            print('iter %d rc:%.5f' % (i, rc.item()))

            if abs(rc-old_rc) < self.tolerance:
                break
            old_rc = rc

        # imageio.mimsave('/export/share/peters57dm/Verbund/deepsync/logs/case_study.gif', imgs, 'GIF', duration=0.35)
        self.labels_ = self.clustering(dist, epsilon)
        return self.labels_

    @staticmethod
    def normalize_feature(data, feature_range=[0, 1]):
        '''
        Adapted from sklearn
        column-wise normalization
        '''
        data_min = torch.min(data.data, dim=0)[0]
        data_max = torch.max(data.data, dim=0)[0]

        data_range = data_max - data_min
        data_range[data_range == 0.0] = 1.0
        scale_ = (feature_range[1] - feature_range[0]) / data_range
        min_ = feature_range[0] - data_min * scale_

        data.data *= scale_
        data.data += min_
        return data

    def euclidean_distances(self, x1, x2):
        '''
        Compute distance map
        Input: x1, x2: [item_num, feature_size]
        Output: [item_num, item_num]
        '''
        x1 = self.normalize_feature(x1)
        x2 = self.normalize_feature(x2)
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)
        delta = x1 - x2
        dist_mtx = (x1 - x2).abs().mean(-1)
        return delta, dist_mtx

    def get_rc(self, dist, neighbor):
        return torch.exp(-dist[neighbor]).mean()

    # def clustering(self, dist, epsilon):  # dist, epsilon=None
    #     # if epsilon is None: epsilon = self.threshold
    #     adjacency = abs(dist.data) < epsilon
    #     num_item = adjacency.shape[0]
    #     clusters, candidate = [], [i for i in range(num_item)]

    #     while candidate:
    #         start_item = candidate.pop(0)
    #         done_nodes = [start_item]
    #         new_nodes = torch.where(adjacency[start_item])[0].int().cpu().numpy().tolist()
    #         if new_nodes:
    #             new_nodes.remove(start_item)
    #         else:
    #             # noise node
    #             clusters.append([start_item])
    #             continue

    #         while new_nodes:
    #             for node in deepcopy(new_nodes):
    #                 done_nodes.append(node)
    #                 adj_nodes = torch.where(adjacency[node])[0].int().cpu().numpy().tolist()
    #                 for adj_node in adj_nodes:
    #                     if adj_node not in done_nodes + new_nodes:
    #                         new_nodes.append(adj_node)

    #                 new_nodes.remove(node)
    #                 candidate.remove(node)
    #         clusters.append(sorted(done_nodes))

    #     print('clustering...')
    #     for idx, cluster in enumerate(clusters):
    #         print(idx, cluster)
    #     print('---------------------------------------------------------------------\n')
    #     return clusters

    def clustering(self, dist, epsilon):
        adjacency = (torch.abs(dist.data) < epsilon)
        num_item = adjacency.shape[0]
        candidate = set(range(num_item))

        # Precompute adjacency list
        adj_list = []
        for i in range(num_item):
            neighbors = torch.where(adjacency[i])[0].tolist()
            if i in neighbors:
                neighbors.remove(i)  # remove self-loop
            adj_list.append(set(neighbors))

        labels = np.full(num_item, -1, dtype=int)  # Initialize all as noise
        cluster_id = 0

        while candidate:
            start_item = candidate.pop()
            neighbors = adj_list[start_item]

            # Noise point: no neighbors
            if not neighbors:
                labels[start_item] = -1
                continue

            # Start new cluster
            cluster = set([start_item])
            labels[start_item] = cluster_id
            queue = deque(neighbors)

            while queue:
                node = queue.popleft()
                if node in candidate:
                    candidate.remove(node)
                    labels[node] = cluster_id
                    cluster.add(node)
                    queue.extend(adj_list[node] - cluster)  # Avoid revisiting

            cluster_id += 1

        return labels


# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#     sync = SyncClus(threshold=0.2, max_iter=100, tolerance=1e-6)

#     # iris = datasets.load_iris()
#     # data = torch.FloatTensor(iris.data)  # .cuda()
#     # label = torch.LongTensor(iris.target)  # .cuda()
#     # trainloader, _ = get_dataLoader(root='../data', dataset='mnist', num_instance=50)
#     # data = torch.FloatTensor(trainloader.dataset.data).view((500, -1)).cuda()
#     # label = torch.LongTensor(trainloader.dataset.targets).cuda()
#     x, y = make_blobs(n_samples=400, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [-0.5, 1.2]],
#                       cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=0)
#     # x, y = make_blobs(n_samples=[200, 40, 200, 200], n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [-0.5, 1.2]],
#     #                   cluster_std=[0.25, 0.7, 0.2, 0.15], random_state=0)
#     # # 生成数据散点图
#     # plt.scatter(x[:, 0], x[:, 1], marker='o')
#     # plt.show()
#     # exit()
#     data, label = torch.FloatTensor(x).cuda(), torch.IntTensor(y).cuda()

#     cluster_label = sync.fit_predict(data, epsilon=0.1)

#     # # time analyse
#     # from line_profiler import LineProfiler
#     # lp = LineProfiler()
#     # lp_wrapper = lp(sync.fit_predict)
#     # lp_wrapper(data, optimizer, metric='euc')
#     # lp.print_stats()
