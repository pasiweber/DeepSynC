import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from sklearn.metrics import pairwise_distances

def find_local_core_points_same(data, k, percent):
    # precomputes the k-th nearest neighbor distance (core_dist / kappa) and reuses this within every T% nearest neighbor subset

    n = data.shape[0]
    subset = int(np.floor(n * percent))

    p_dist = pairwise_distances(data, metric="euclidean")
    core_dists = np.partition(p_dist, k - 1, axis=0)[k - 1]

    nn = np.argpartition(p_dist, subset, axis=1)[:, :subset]

    refined_medians = np.median(core_dists[nn], axis=1)
    med_th = np.median(refined_medians)

    vector_mask = core_dists < refined_medians
    core_points_mask = np.outer(vector_mask, vector_mask).astype(int)
    return core_points_mask, med_th

def _sync_loss(model, batch_data, current_batch_labels, device):
    def get_scaled_outlier_dists(out_dists):
        # input: distance matrix of a batch masked for the distances of outliers
        # output: scaled distances for outliers based on distance
        max_vec = np.max(out_dists, 1)
        s = max_vec / 4
        s = np.transpose(np.repeat([s], out_dists.shape[0], axis=0))
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

    return loss, losses_tracker

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

def knn_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k=25):

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

def not_assigning_new_points(evaltracker, n):
    def all_equal(lst):
        return all(x == lst[0] for x in lst)

    number_labeled_points = evaltracker.no_labeled_pts[-n:]
    return all_equal(number_labeled_points)