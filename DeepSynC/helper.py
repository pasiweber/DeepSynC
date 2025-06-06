import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from scipy.spatial.distance import cdist
from SHiP import SHiP
import sys

sys.path.append("/export/share/peters57dm/Verbund/deepsync/experiments/")
from helper.tracker import update_eval_tracker
from helper.deep import encode_batchwise
from helper.utils import load_json_as_dict, save_dict_as_json
from helper.plots import plot_2d_dataset, create_gif_from_directory

def get_eps_neighbourhood(data, eps):
    n = data.shape[0]
    squared_diffs = (data.unsqueeze(0) - data.unsqueeze(1)).pow(2).sum(2)
    # NB_eps_mask = torch.Tensor(np.double(squared_diffs < eps)-np.eye(n)) # exclude point itself
    NB_eps_mask = torch.Tensor(np.double(squared_diffs < eps))  # include point becuase empty neighbourhood produces nan
    return squared_diffs, NB_eps_mask


# Kuramoto Order Parameter with eps neighbourhood
def calc_KOP_eps(data, eps):
    n = data.shape[0]
    di, mask = get_eps_neighbourhood(data, eps)
    r_c = 1 / n * (1 / (mask.sum(0)) * (np.exp(-di) * mask).sum(0)).sum()
    return r_c


# distances to k nearest neighbours and their indices
def get_k_nns(data, k):
    squared_diffs = (data.unsqueeze(0) - data.unsqueeze(1)).pow(2).sum(2)
    knn_distance_based = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(squared_diffs)
    distances, indices = knn_distance_based.kneighbors(squared_diffs)
    return distances, indices


# Kuramoto Order Parameter with k NNs
def calc_KOP_nn(data, k):
    n = data.shape[0]
    dists, k_nn_ind = get_k_nns(data, k)
    r_c = np.exp(-dists).sum()
    r_c *= 1 / (n * k)
    return r_c


def calculate_squared_differences_vectorized(data):
    """
    Function to calculate the squared differences between all pairs of data points in a vectorized manner.

    Parameters:
    - data: A numpy array of shape (n_samples, n_features), representing the dataset.

    Returns:
    - squared_diff_matrix: A (n_samples, n_samples) numpy array containing the squared differences between all pairs.
    """
    squared_diffs = (data.unsqueeze(0) - data.unsqueeze(1)).pow(2).sum(2)

    return squared_diffs.detach().cpu().numpy()


def find_core_points(squared_diffs, k):
    """
    Function to find core points in regions of high density.

    Parameters:
    - data: A numpy array of shape (n_samples, n_features), representing the dataset.
    - eps: The distance within which to consider neighbors.
    - min_samples: The minimum number of points required to be considered a core point.

    Returns:
    - core_points: A numpy array containing the actual coordinates of the core points.
    """
    # Fit NearestNeighbors to find neighbors within eps distance
    n = len(squared_diffs)
    knn_distance_based = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(squared_diffs)
    distances, indices = knn_distance_based.kneighbors(squared_diffs)
    density_threshold = np.median(distances[:, k - 1])
    mask = np.ones((n, k))
    mask[distances[:, k - 1] > density_threshold, :] = 0

    core_mask_cpu = np.zeros((n, n))
    for j in range(0, n):
        for l in range(0, n):
            if mask[j, 0] == 1 and mask[l, 0] == 1:
                core_mask_cpu[j, l] = 1
    # final_mask_cpu = core_mask_cpu*(squared_diffs<(density_threshold))
    return core_mask_cpu


# precomputes the k-th nearest neighbor distance (core_dist / kappa) and reuses this within every T% nearest neighbor subset
def find_local_core_points_same(data, k, percent):
    from sklearn.metrics import pairwise_distances

    n = data.shape[0]
    subset = int(np.floor(n * percent))

    p_dist = pairwise_distances(data, metric="euclidean")
    core_dists = np.partition(p_dist, k - 1, axis=0)[k - 1]

    nn = np.argpartition(p_dist, subset, axis=1)[:, :subset]
    # nn_mask = np.tile(np.arange(n).reshape(-1, 1), (1, subset))
    # nn = nn[nn != nn_mask].reshape(n, subset - 1)
    refined_medians = np.median(core_dists[nn], axis=1)
    med_th = np.median(refined_medians)

    vector_mask = core_dists < refined_medians
    core_points_mask = np.outer(vector_mask, vector_mask).astype(int)
    return core_points_mask, med_th


def find_local_core_points_fast(data, k, percent):
    n = data.shape[0]
    subset = int(np.floor(n * percent))

    knn = NearestNeighbors(n_neighbors=subset, metric="euclidean").fit(data)
    distances, indices = knn.kneighbors(data)

    # Compute k-nearest neighbors within each subset
    refined_medians = np.zeros(n)
    for j in range(n):
        j_neighbors = data[indices[j], :]
        knn_j = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(j_neighbors)
        distances_j, _ = knn_j.kneighbors(j_neighbors)
        refined_medians[j] = np.median(distances_j[:, k - 1])

    furthest_neighbor_points_distances = distances[:, k - 1]
    vector_mask = furthest_neighbor_points_distances < refined_medians

    core_points_mask = np.outer(vector_mask, vector_mask).astype(int)
    core_threshold = np.median(refined_medians)

    return core_points_mask, core_threshold


def find_local_core_points(data, k, percent):
    # Fit NearestNeighbors to find neighbors within eps distance
    n = data.shape[0]
    subset = int(np.floor(n * percent))
    knn_distance_based = NearestNeighbors(n_neighbors=subset, metric="euclidean").fit(data)
    distances, indices = knn_distance_based.kneighbors(data)
    mask = np.zeros(n)
    medians = np.zeros(n)
    for j in range(0, n):
        j_neighbours = data[indices[j, :], :]
        knn_distance_based_j = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(j_neighbours)
        distances_j, _ = knn_distance_based_j.kneighbors(j_neighbours)
        density_threshold = np.median(distances_j[:, k - 1])
        medians[j] = density_threshold
        if distances[j, k - 1] < density_threshold:  # or distances_j[0,k-1]
            mask[j] = 1
    # x = np.where(mask==1)
    # core_point_ind= x[0]
    core_mask_cpu = np.zeros((n, n))
    for r in range(0, n):
        for l in range(0, n):
            if mask[r] == 1 and mask[l] == 1:
                core_mask_cpu[r, l] = 1
    core_thresh = np.median(medians)
    return core_mask_cpu, core_thresh


# endregion


# region Loss Functions
def attraction_repelling_loss(model, batch_data, iteration_labels, losses_tracker, device):
    embedded = model.encode(batch_data)
    unique_labels = np.unique(iteration_labels)
    unique_labels = unique_labels[unique_labels > -1]
    if len(unique_labels) == 0:
        print("No labels in this batch")
        dont_propagate = True
        loss = 0
        return loss, dont_propagate, losses_tracker

    dont_propagate = False
    attract_loss = 0
    repel_loss = 0
    for l in unique_labels:
        # calculate attraction loss
        label_mask = iteration_labels == l
        label_pts = embedded[label_mask]
        label_square_diffs = (label_pts.unsqueeze(0) - label_pts.unsqueeze(1)).pow(2).sum(2)
        label_weights = 1 - (label_square_diffs / torch.max(label_square_diffs))
        n_label_pts = len(label_pts)
        _att_val = 1 / n_label_pts**2 * torch.exp(-label_square_diffs * label_weights).sum()
        if not torch.isnan(_att_val):
            attract_loss += 1 - _att_val

        # calculate repeling loss
        other_pts_mask = np.logical_and(iteration_labels != l, iteration_labels > -1)
        other_pts = embedded[other_pts_mask]
        other_pts_square_diffs = (label_pts.unsqueeze(0) - other_pts.unsqueeze(1)).pow(2).sum(2)
        label_weights = 1  # fixed label weight
        no_other_pts = len(other_pts_square_diffs)
        if no_other_pts == 0:
            _rep_val = torch.tensor(0)
        else:
            _rep_val = 1 / (no_other_pts) ** 2 * torch.exp(-other_pts_square_diffs * label_weights).sum()

        if not torch.isnan(_rep_val):
            repel_loss += _rep_val

    loss = 1 / len(unique_labels) * (attract_loss + repel_loss)

    if not isinstance(loss, int):
        losses_tracker.total.append(loss.detach().cpu().numpy().item())
    else:
        losses_tracker.total.append(loss)

    if not isinstance(repel_loss, int):
        losses_tracker.repel.append(repel_loss.detach().cpu().numpy().item())
    else:
        losses_tracker.repel.append(repel_loss)

    if not isinstance(attract_loss, int):
        losses_tracker.attract.append(attract_loss.detach().cpu().numpy().item())
    else:
        losses_tracker.attract.append(attract_loss)

    return loss, dont_propagate, losses_tracker


def _sync_loss(model, batch_data, current_batch_labels, device):
    def get_scaled_outlier_dists(out_dists):
        # input: distance matrix of a batch masked for the distances of outliers
        # output: scaled distances for outliers based on distance
        max_vec = np.max(out_dists, 1)
        s = max_vec / 4
        s = np.transpose(np.repeat([s], out_dists.shape[0], axis=0))
        # print(s)
        x = np.multiply(out_dists, 1 / s)
        return np.exp(-1 / 2 * np.power(x, 2))

    embedded = model.encode(batch_data)
    squared_diffs = (embedded.unsqueeze(0) - embedded.unsqueeze(1)).pow(2).sum(2)
    squared_diffs_cpu = squared_diffs.detach().cpu().numpy()
    n = len(current_batch_labels)
    outliers = np.eye(n)
    for j in range(0, n):
        if current_batch_labels[j] < 0:
            outliers[:, j] = 1
            outliers[j, :] = 1
    outlier_dists = squared_diffs_cpu * outliers
    scaled_outlier_weights = get_scaled_outlier_dists(outlier_dists)
    scaled_weights = scaled_outlier_weights
    for k in range(0, n):
        if current_batch_labels[k] >= 0:
            for l in range(0, n):
                if current_batch_labels[l] >= 0:
                    if current_batch_labels[k] == current_batch_labels[l]:
                        scaled_weights[k, l] = 1
                        scaled_weights[l, k] = 1
                    else:
                        scaled_weights[k, l] = 0
                        scaled_weights[l, k] = 0
    scaled_weights_torch = torch.from_numpy(scaled_weights).to(device)
    sync = (1 / n**2) * (torch.exp(-squared_diffs * scaled_weights_torch)).sum(0).sum()
    sync_loss = 1 - sync
    return sync_loss


def _reconstrunction_mse_loss(model, batch_data):
    embedded = model.encode(batch_data)
    reconstruction = model.decode(embedded)
    loss_fun = torch.nn.MSELoss()
    ae_loss = loss_fun(reconstruction, batch_data)
    return ae_loss


def ae_sync_loss(model, batch_data, current_batch_labels, losses_tracker, device):
    ae_loss = _reconstrunction_mse_loss(model, batch_data)
    sync_loss = _sync_loss(model, batch_data, current_batch_labels, device)
    loss = ae_loss + sync_loss
    losses_tracker.ae_loss.append(ae_loss.detach().cpu().numpy().item())
    losses_tracker.sync_loss.append(sync_loss.detach().cpu().numpy().item())
    losses_tracker.total.append(loss.detach().cpu().numpy().item())
    dont_propagate = False
    return loss, dont_propagate, losses_tracker


# endregion


# region assign labels
def get_high_cof_labels(prediction_matrix, n, crop):
    cols_to_check = prediction_matrix
    if crop is not None:
        cols_to_check = prediction_matrix[:, -n:]
    else:  # iteration number < n_check
        # initialize with all zeros
        high_confidence_labels = np.zeros(prediction_matrix.shape[0], dtype=bool)
        # all labeled points are high confidence labels
        high_confidence_labels[cols_to_check[:, 0] >= 0] = True
        return high_confidence_labels
    i = 0
    all_equal = cols_to_check[:, i] == cols_to_check[:, i + 1]
    while i + 1 < cols_to_check.shape[1]:
        check = cols_to_check[:, i] == cols_to_check[:, i + 1]
        all_equal = np.logical_and(all_equal, check)
        i += 1
    custom_labels_mask = prediction_matrix[:, -2] < 0
    all_equal[custom_labels_mask] = False
    return all_equal


def check_equality_in_n_consequtive_cols(prediction_matrix, n, crop):
    cols_to_check = prediction_matrix
    if crop is not None:
        cols_to_check = prediction_matrix[:, crop - n : crop]
    i = 0
    all_equal = cols_to_check[:, i] == cols_to_check[:, i + 1]
    while i + 1 < cols_to_check.shape[1]:
        check = cols_to_check[:, i] == cols_to_check[:, i + 1]
        all_equal = np.logical_and(all_equal, check)
        i += 1
    custom_labels_mask = prediction_matrix[:, i] < 0
    all_equal[custom_labels_mask] = False
    return all_equal


def most_frequent(arr):
    n = len(arr)
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if max_count < Hash[i]:
            res = i
            max_count = Hash[i]

    return res


def find_cluster_intersections(labels1, labels2):
    # Check if the two lists are of the same length
    if len(labels1) != len(labels2):
        raise ValueError("Both label lists must be of the same length")

    # Dictionary to store elements of each cluster for both label lists
    clusters1 = defaultdict(list)
    clusters2 = defaultdict(list)

    # Group elements by their cluster labels in labels1
    for idx, label in enumerate(labels1):
        clusters1[label].append(idx)

    # Group elements by their cluster labels in labels2
    for idx, label in enumerate(labels2):
        clusters2[label].append(idx)

    # Find intersections between clusters
    intersections = []
    for cluster1, elements1 in clusters1.items():
        for cluster2, elements2 in clusters2.items():
            # Find common elements between two clusters
            common_elements = set(elements1).intersection(elements2)
            if common_elements:
                intersections.append((cluster1, cluster2, list(common_elements)))

    return intersections


def assign_unified_labels(labels1, labels2):
    intersections = find_cluster_intersections(labels1, labels2)

    # Create a unified label list initialized with -1 (or any placeholder for unassigned)
    unified_labels = [-1] * len(labels1)

    # Assign a new unified label to each intersection
    current_label = 0
    for cluster1, cluster2, common_elements in intersections:
        if cluster1 == -1 or cluster2 == -1:
            continue
        for idx in common_elements:
            unified_labels[idx] = current_label
        current_label += 1
    unified_labels = np.array(unified_labels)
    for label in set(unified_labels):
        if label == -1:
            continue
        mask = unified_labels == label
        mask.astype(int)
        if sum(mask) == 1:
            # assign -1 (outlier) if labels disagree
            unified_labels[mask] = -1
    final_label_mapping = {-1: -1}
    label_max = -1
    for label in set(unified_labels):
        if not label in final_label_mapping:
            label_max += 1
            final_label_mapping[label] = label_max

    # # allow change in labels if previously was defined as outlier
    # unified_labels[labels1 == -1] = labels2[labels1 == -1]
    # # if a point is considered an outlier while it was not before, ignore it.
    # unified_labels[labels2 == -1] = labels1[labels2 == -1]

    # final labeling assignment
    for k, v in final_label_mapping.items():
        if k == -1:
            unified_outlier_mask = unified_labels == k
            labels2_outlier_mask = labels2 == k
            labels1_outlier_mask = labels1 == k

            # In any case
            # it must be an outlier in the unified labels final results

            # case_0: if outlier in labels1 and labels2 => set as outlier
            case_0_mask = np.logical_and(labels1_outlier_mask, labels2_outlier_mask)
            # Ensure that those indicies are already classified as outliers
            # in the unified labels
            case_0_mask = np.logical_and(case_0_mask, unified_outlier_mask)
            unified_labels[case_0_mask] = -1

            # case_1: if outlier in labels2 only => set label of labels1
            case_1_mask = np.logical_and(labels2_outlier_mask == True, labels1_outlier_mask == False)
            # Ensure that those indicies are already classified as outliers
            # in the unified labels
            case_1_mask = np.logical_and(case_1_mask, unified_outlier_mask)

            # assign the most frequent label
            # This trick is done because we can't assume that
            # the label number is the same for each cluster
            # across different iterations
            case_1_labels1_possible_values = labels1[case_1_mask]
            case_1_labels1_most_frequent = most_frequent(case_1_labels1_possible_values)
            case_1_labels1_most_frequent_mask = labels1 == case_1_labels1_most_frequent

            possible_values = unified_labels[case_1_labels1_most_frequent_mask]
            value = most_frequent(possible_values)
            unified_labels[case_1_mask] = value

            # case_2: if outlier in labels1 only => set label of labels2
            case_2_mask = np.logical_and(labels1_outlier_mask == True, labels2_outlier_mask == False)

            # Ensure that those indicies are already classified as outliers
            # in the unified labels
            case_2_mask = np.logical_and(case_2_mask, unified_outlier_mask)

            # assign the most frequent label
            case_2_labels2_possible_values = labels2[case_2_mask]
            case_2_labels2_most_frequent = most_frequent(case_2_labels2_possible_values)
            case_2_labels2_most_frequent_mask = labels2 == case_2_labels2_most_frequent

            possible_values = unified_labels[case_2_labels2_most_frequent_mask]
            value = most_frequent(possible_values)
            unified_labels[case_2_mask] = value

            # case_3: if not outlier in any of them set label of labels2
            case_3_mask = np.logical_and(labels1_outlier_mask == False, labels2_outlier_mask == False)

            # Ensure that those indicies are already classified as outliers
            # in the unified labels
            case_3_mask = np.logical_and(case_3_mask, unified_outlier_mask)

            # assign the most frequent label
            case_3_labels2_possible_values = labels2[case_3_mask]
            case_3_labels2_most_frequent = most_frequent(case_3_labels2_possible_values)
            case_3_labels2_most_frequent_mask = labels2 == case_3_labels2_most_frequent

            possible_values = unified_labels[case_3_labels2_most_frequent_mask]
            value = most_frequent(possible_values)
            unified_labels[case_3_mask] = value

        # assign the unified labels
        mask = unified_labels == k
        unified_labels[mask] = v

    return unified_labels


def get_high_conf_labels(prediction_matrix, n_check, iter_n):
    cols_to_check = prediction_matrix.copy()
    crop = iter_n < n_check
    if crop:
        cols_to_check = cols_to_check[:, :iter_n]
    else:
        cols_to_check = cols_to_check[:, iter_n - n_check : iter_n]
    i = 0
    # handle boundary conditions
    if i + 1 < cols_to_check.shape[1]:
        all_equal = cols_to_check[:, i] == cols_to_check[:, i + 1]
    else:
        all_equal = cols_to_check[:, i] > -1
    # if labels disagree in second epoch, it is not reliable enough
    while i + 1 < cols_to_check.shape[1]:
        check = cols_to_check[:, i] == cols_to_check[:, i + 1]
        all_equal = np.logical_and(all_equal, check)
        i += 1
    custom_labels_mask = cols_to_check[:, -1] < 0
    all_equal[custom_labels_mask] = False
    return all_equal


def distance_nearest_neighbor(labeled_points, labels, unlabeled_points, distance_threshold):
    # in future implementation, we need to include the option of mahalanobis distance from
    # the centroid of each cluster in the labeled points.
    """
    Assign labels to unlabeled points based on the nearest labeled points
    while keeping far points unlabeled.

    Parameters:
    - labeled_points: np.ndarray of shape (n_labeled, n_features)
    - labels: np.ndarray of shape (n_labeled,)
    - unlabeled_points: np.ndarray of shape (n_unlabeled, n_features)
    - distance_threshold: float, the threshold distance for assigning labels
    Returns:
    - assigned_labels: np.ndarray of shape (n_unlabeled,), assigned labels for unlabeled points
    """
    # Compute distances from unlabeled points to labeled points
    distances = cdist(unlabeled_points, labeled_points)
    ###
    # we can use the mahalanobis distance here to consider the data distribution in the calc
    ###

    # Initialize an array to store assigned labels
    assigned_labels = np.full(unlabeled_points.shape[0], -1)  # -1 for unlabeled
    # Assign labels to the nearest labeled points if within the threshold
    for i in range(distances.shape[0]):
        nearest_index = np.argmin(distances[i])  # Index of the nearest labeled point
        nearest_distance = distances[i, nearest_index]

        if nearest_distance < distance_threshold:
            assigned_labels[i] = labels[nearest_index]
    return assigned_labels


def calc_distance_threshold(data_complete, all_labels):
    mask = all_labels > -1
    labels = all_labels[mask]
    data = data_complete[mask]
    labels_set = set(labels)
    distance_medians = []
    for label in labels_set:
        cluster_mask = labels == label
        cluster_pts = data[cluster_mask]
        squared_diffs = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2)
        cluster_median_distance = np.median(squared_diffs)
        distance_medians.append(cluster_median_distance)
    return np.min(distance_medians) * 0.05


def mahalanobis_assign_unlabeled_points(data, labels):
    from scipy.spatial.distance import mahalanobis
    from scipy.spatial.distance import euclidean

    unique_labels = set(labels) - {-1}  # Get unique cluster labels excluding -1

    # Compute cluster centroids, covariance matrices, and thresholds
    centroids = {}
    cov_matrices = {}
    inv_cov_matrices = {}
    thresholds = {}

    outlier_points = data[labels == -1]
    if len(outlier_points) == 0:  # all points are labeled
        return labels  # return the same labels

    for label in unique_labels:
        cluster_points = data[labels == label]
        centroids[label] = np.mean(cluster_points, axis=0)
        cov_matrices[label] = np.cov(cluster_points, rowvar=False)

        if cov_matrices[label].ndim < 2:  # to handle zero dimensional arrays when no points for a certain label.
            thresholds[label] = 0
            continue
        # Compute inverse covariance matrix (handle singular matrices)
        try:
            inv_cov_matrices[label] = np.linalg.inv(cov_matrices[label])
        except np.linalg.LinAlgError:
            inv_cov_matrices[label] = np.linalg.pinv(cov_matrices[label])  # Use pseudo-inverse if singular

        # Compute the Mahalanobis distance for each point in the cluster to its centroid
        distances = [mahalanobis(p, centroids[label], inv_cov_matrices[label]) for p in outlier_points]
        thresholds[label] = np.percentile(distances, 10)
        # distances = [mahalanobis(p, centroids[label], inv_cov_matrices[label]) for p in cluster_points]
        # thresholds[label] = np.max(distances) * 1.05

    # Assign unlabeled points (-1) to the nearest centroid based on Mahalanobis distance
    for i, point in enumerate(data):
        if labels[i] == -1:
            min_distance = float("inf")
            assigned_label = -1

            for label in unique_labels:
                try:
                    distance = mahalanobis(point, centroids[label], inv_cov_matrices[label])
                except (
                    KeyError
                ):  # couldn't calculate inv covariance of this label, we can also use euclidean instead of mahalanobis!.
                    distance = float("inf")
                if distance < min_distance:
                    min_distance = distance
                    assigned_label = label

            # Assign only if within the calculated threshold for that cluster
            if assigned_label != -1 and min_distance <= thresholds[assigned_label]:
                labels[i] = assigned_label

    return labels


def euclidean_assign_unlabeled_points(data, labels):
    from scipy.spatial.distance import euclidean

    unique_labels = set(labels) - {-1}  # Get unique cluster labels excluding -1

    # Compute cluster centroids
    centroids = {label: np.mean(data[labels == label], axis=0) for label in unique_labels}

    outlier_points = data[labels == -1]
    if len(outlier_points) == 0:  # All points are labeled
        return labels  # Return the same labels

    # Compute distance thresholds based on the 10th percentile of distances
    thresholds = {}
    for label in unique_labels:
        cluster_points = data[labels == label]
        distances = [euclidean(p, centroids[label]) for p in cluster_points]
        thresholds[label] = np.percentile(distances, 10)

    # Assign unlabeled points (-1) to the nearest centroid based on Euclidean distance
    for i, point in enumerate(data):
        if labels[i] == -1:
            min_distance = float("inf")
            assigned_label = -1

            for label in unique_labels:
                distance = euclidean(point, centroids[label])
                if distance < min_distance:
                    min_distance = distance
                    assigned_label = label

            # Assign only if within the calculated threshold for that cluster
            if assigned_label != -1 and min_distance <= thresholds[assigned_label]:
                labels[i] = assigned_label

    return labels


def knn_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k=25):
    from scipy.stats import mode

    unlabeled_points_mask = current_epoch_labels < 0
    unlabeled_points = train_embedded_data[unlabeled_points_mask, :]
    if len(unlabeled_points) == 0:
        return current_epoch_labels
    knn_for_labelling = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embedded_data)
    indices_for_labelling = knn_for_labelling.kneighbors(unlabeled_points, return_distance=False)
    labels_of_neighbours = current_epoch_labels[indices_for_labelling]
    most_common_labels = mode(labels_of_neighbours, axis=1, keepdims=False)
    new_labels = most_common_labels.mode
    current_epoch_labels[unlabeled_points_mask] = new_labels
    return current_epoch_labels


def knn_average_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k=25):
    unlabeled_points_mask = current_epoch_labels < 0
    unlabeled_points = train_embedded_data[unlabeled_points_mask, :]
    if len(unlabeled_points) == 0:
        return current_epoch_labels

    # Only fit on labeled data
    labeled_points_mask = current_epoch_labels >= 0
    labeled_data = train_embedded_data[labeled_points_mask, :]
    labeled_labels = current_epoch_labels[labeled_points_mask]

    # kNN fit on labeled data only
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(labeled_data)
    distances, indices = knn.kneighbors(unlabeled_points, return_distance=True)

    new_labels = []

    for i in range(len(unlabeled_points)):
        cluster_distance_sum = defaultdict(list)

        for j in range(k):
            label = labeled_labels[indices[i][j]]
            distance = distances[i][j]
            cluster_distance_sum[label].append(distance)

        # Compute average distance to each cluster among k nearest neighbors
        avg_distances = {label: np.mean(dists) for label, dists in cluster_distance_sum.items()}

        # Assign the label with the smallest average distance
        best_label = min(avg_distances, key=avg_distances.get)
        new_labels.append(best_label)

    current_epoch_labels[unlabeled_points_mask] = new_labels
    return current_epoch_labels


def vote_of_two_knn_methods(train_embedded_data, current_epoch_labels, k=25):
    label1 = knn_average_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k)
    label2 = knn_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k)
    same_mask = label1 == label2
    combined_label = np.copy(label1)
    combined_label[~same_mask] = -1
    return combined_label


# endregion


# region DeepSync
def not_assigning_new_points(evaltracker, n):
    def all_equal(lst):
        return all(x == lst[0] for x in lst)

    number_labeled_points = evaltracker.no_labeled_pts[-n:]
    return all_equal(number_labeled_points)


def get_local_core_points(data, k, percent, path):
    if os.path.exists(path):
        core_threshold = None
        return np.load(path).astype(int), core_threshold
    core_points_mask, core_threshold = find_local_core_points_fast(data, k, percent)
    np.save(path, core_points_mask.astype(bool))
    return core_points_mask, core_threshold


def deep_sync_model_gtlabels(
    device,
    model,
    data,
    dataset_name,
    trainloader,
    testloader,
    optimizer,
    n_check,
    k,
    percent,
    training_iterations,
    loss_fn,
    labels_assignment_method,
    losses_tracker,
    eval_tracker,
    directory,
    deep_sync_model_path,
):
    # Create folder to save results in
    imgs_directory = os.path.join(directory, "images")
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(trackers_path, exist_ok=True)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    labels_over_iterations = np.zeros((len(data), training_iterations + 1)) - 10
    core_points_mask, th = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    original_labels[np.where(np.diag(core_points_mask) == 1)[0]] = gt_labels[
        np.where(np.diag(core_points_mask) == 1)[0]
    ]
    labels_over_iterations[:, 0] = original_labels
    i = 0
    while i < training_iterations:
        for batch, batch_labels, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, dont_propagate, losses_tracker = loss_fn(model, batch_data, iteration_labels, losses_tracker, device)
            if dont_propagate:
                continue
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        # label assignment
        current_epoch_labels = labels_over_iterations[:, i]
        current_epoch_labels = labels_assignment_method(train_embedded_data, current_epoch_labels)
        if i > n_check:
            crop = i
        else:
            crop = None
        high_confidence_labels = check_equality_in_n_consequtive_cols(labels_over_iterations, n_check, crop)
        previous_epoch_labels = labels_over_iterations[:, i]
        unified_labels = assign_unified_labels(previous_epoch_labels, current_epoch_labels)
        unified_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]
        # 4 - Store the labels
        labels_over_iterations[:, i + 1] = unified_labels
        eval_tracker = update_eval_tracker(gt_labels, unified_labels, eval_tracker)
        saving_path = os.path.join(imgs_directory, "{0:03}".format(i) + ".jpg")
        plot_2d_dataset(train_embedded_data, unified_labels, centers=None, fixed_scales=False, save=saving_path)
        i += 1
        if np.all(high_confidence_labels):
            print(f"apply early stopping after {i} iterations, all points are labeled confidently.")
            break
    # Create and save the iterations Plots GIF
    create_gif_from_directory(imgs_directory, os.path.join(directory, f"{dataset_name}.gif"), duration=300)
    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(), os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(), os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations


def deep_sync_model_ship(
    device,
    model,
    data,
    dataset_name,
    trainloader,
    testloader,
    optimizer,
    n_check,
    k,
    percent,
    training_iterations,
    loss_fn,
    labels_assignment_method,
    losses_tracker,
    eval_tracker,
    directory,
    deep_sync_model_path,
):
    # Create folder to save results in
    imgs_directory = os.path.join(directory, "images")
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(trackers_path, exist_ok=True)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    true_k = len(torch.unique(gt_labels))
    labels_over_iterations = np.zeros((len(data), 1))
    core_points_mask, th = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    core_points = embedded[np.where(np.diag(core_points_mask) == 1)[0]]
    ship = SHiP(data=core_points, treeType="DCTree")
    ship_labels = ship.fit_predict(power=2, partitioningMethod="ThreshholdElbow")
    original_labels[np.where(np.diag(core_points_mask) == 1)[0]] = torch.FloatTensor(ship_labels)
    eval_tracker = update_eval_tracker(gt_labels, original_labels.numpy(), eval_tracker)

    print("Initial clustering evalution:")
    print("AMI = ", eval_tracker.ami_labeled)
    print("ARI = ", eval_tracker.ari_labeled)
    print(
        f"labeled points = {eval_tracker.no_labeled_pts}/{len(eval_tracker.predicted_labels)}",
    )
    print("-------------------------------------")

    labels_over_iterations[:, 0] = original_labels
    i = 0
    while i < training_iterations:
        for batch, batch_labels, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, dont_propagate, losses_tracker = loss_fn(model, batch_data, iteration_labels, losses_tracker, device)
            if dont_propagate:
                continue
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        # label assignment
        if i >= n_check:
            crop = i
        else:
            crop = None
        high_confidence_labels = get_high_cof_labels(labels_over_iterations, n_check, crop)
        current_epoch_labels = labels_over_iterations[:, i]
        # consider points with low confidence as -1 so that they can be assigned to other clusters
        current_epoch_labels[~high_confidence_labels] = -1
        current_epoch_labels = labels_assignment_method(train_embedded_data, current_epoch_labels)
        previous_epoch_labels = labels_over_iterations[:, i]
        # unified_labels = assign_unified_labels(
        #     previous_epoch_labels,
        #     current_epoch_labels
        # )

        # prevent high confidence labels from changing
        current_epoch_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]
        # prevent labeled points from getting unlabeled
        labeled_to_unlabeled_points_mask = np.logical_and(current_epoch_labels == -1, previous_epoch_labels != -1)
        current_epoch_labels[labeled_to_unlabeled_points_mask] = previous_epoch_labels[labeled_to_unlabeled_points_mask]

        # 4 - Store the labels
        labels_over_iterations = np.hstack(
            (labels_over_iterations, current_epoch_labels.reshape(-1, 1))
        )  # let the labels over iterations matrix grow
        eval_tracker = update_eval_tracker(gt_labels, current_epoch_labels, eval_tracker)
        saving_path = os.path.join(imgs_directory, "{0:03}".format(i) + ".jpg")
        plot_2d_dataset(train_embedded_data, current_epoch_labels, centers=None, fixed_scales=False, save=saving_path)
        i += 1

        # Early Stopping
        if np.all(high_confidence_labels):
            print(f"Early stopping after {i} iterations, all points are labeled confidently.")
            break
        if not_assigning_new_points(eval_tracker, n_check):
            print(
                f"Early stopping after {i} iterations, the algorithm is not assigning any new points for {n_check} iterations."
            )
            break

    # Create and save the iterations Plots GIF
    create_gif_from_directory(imgs_directory, os.path.join(directory, f"{dataset_name}.gif"), duration=300)
    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(), os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(), os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations


def deep_sync_model_ship_trueK(
    device,
    model,
    data,
    dataset_name,
    trainloader,
    testloader,
    optimizer,
    n_check,
    k,
    percent,
    training_iterations,
    loss_fn,
    labels_assignment_method,
    losses_tracker,
    eval_tracker,
    directory,
    deep_sync_model_path,
):
    # Create folder to save results in
    imgs_directory = os.path.join(directory, "images")
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(trackers_path, exist_ok=True)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    true_k = len(torch.unique(gt_labels))
    labels_over_iterations = np.zeros((len(data), 1))
    core_points_mask, th = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    core_points = embedded[np.where(np.diag(core_points_mask) == 1)[0]]
    ship = SHiP(data=core_points, treeType="DCTree", config={"k": true_k})
    ship_labels = ship.fit_predict(power=2, partitioningMethod="K")
    original_labels[np.where(np.diag(core_points_mask) == 1)[0]] = torch.FloatTensor(ship_labels)
    eval_tracker = update_eval_tracker(gt_labels, original_labels.numpy(), eval_tracker)

    print("Initial clustering evalution:")
    print("AMI = ", eval_tracker.ami_labeled)
    print("ARI = ", eval_tracker.ari_labeled)
    print(
        f"labeled points = {eval_tracker.no_labeled_pts}/{len(eval_tracker.predicted_labels)}",
    )
    print("-------------------------------------")

    labels_over_iterations[:, 0] = original_labels
    i = 0
    while i < training_iterations:
        for batch, batch_labels, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, dont_propagate, losses_tracker = loss_fn(model, batch_data, iteration_labels, losses_tracker, device)
            if dont_propagate:
                continue
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        # label assignment
        if i >= n_check:
            crop = i
        else:
            crop = None
        high_confidence_labels = get_high_cof_labels(labels_over_iterations, n_check, crop)
        current_epoch_labels = labels_over_iterations[:, i]
        # consider points with low confidence as -1 so that they can be assigned to other clusters
        current_epoch_labels[~high_confidence_labels] = -1
        current_epoch_labels = labels_assignment_method(train_embedded_data, current_epoch_labels)
        previous_epoch_labels = labels_over_iterations[:, i]
        # unified_labels = assign_unified_labels(
        #     previous_epoch_labels,
        #     current_epoch_labels
        # )

        # prevent high confidence labels from changing
        current_epoch_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]
        # prevent labeled points from getting unlabeled
        labeled_to_unlabeled_points_mask = np.logical_and(current_epoch_labels == -1, previous_epoch_labels != -1)
        current_epoch_labels[labeled_to_unlabeled_points_mask] = previous_epoch_labels[labeled_to_unlabeled_points_mask]

        # 4 - Store the labels
        labels_over_iterations = np.hstack(
            (labels_over_iterations, current_epoch_labels.reshape(-1, 1))
        )  # let the labels over iterations matrix grow
        eval_tracker = update_eval_tracker(gt_labels, current_epoch_labels, eval_tracker)
        saving_path = os.path.join(imgs_directory, "{0:03}".format(i) + ".jpg")
        plot_2d_dataset(train_embedded_data, current_epoch_labels, centers=None, fixed_scales=False, save=saving_path)
        i += 1

        # Early Stopping
        if np.all(high_confidence_labels):
            print(f"Early stopping after {i} iterations, all points are labeled confidently.")
            break
        if not_assigning_new_points(eval_tracker, n_check):
            print(
                f"Early stopping after {i} iterations, the algorithm is not assigning any new points for {n_check} iterations."
            )
            break

    # Create and save the iterations Plots GIF
    create_gif_from_directory(imgs_directory, os.path.join(directory, f"{dataset_name}.gif"), duration=300)
    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(), os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(), os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations


# endregion
