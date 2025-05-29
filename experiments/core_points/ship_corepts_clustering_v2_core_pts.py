import os
import sys
import numpy as np

sys.path.append("/export/share/peters57dm/Verbund/deepsync/experiments/")
from helper.datasets import (
    load_pendigits,
    load_optdigits,
    load_letterrecognition,
    load_gaussian_blobs,
    load_example,
    load_usps,
    load_htru,
    load_har,
    load_mice,
    load_synth_high,
    load_synth_low,
    load_mnist,
    load_fmnist,
    load_cifar10,
    load_coil20,
    load_coil100,
    load_cifar100,
    load_weizmann,
)
from helper.deep import (
    Autoencoder,
    get_train_and_testloader,
    load_pretrained_model,
    encode_batchwise,
)
from sklearn.metrics import pairwise_distances
from SHiP import SHiP
from SHiP.ultrametric_tree import UltrametricTreeType as UTreeType, AVAILABLE_ULTRAMETRIC_TREE_TYPES
from SHiP.partitioning import PartitioningMethod as PMethod, AVAILABLE_PARTITIONING_METHODS


datasets_loading_methods = [
    load_pendigits,
    load_optdigits,
    load_letterrecognition,
    load_gaussian_blobs,
    load_example,
    load_usps,
    load_htru,
    load_har,
    load_mice,
    load_synth_high,
    load_synth_low,
    load_mnist,
    load_fmnist,
    load_cifar10,
    load_coil20,
    load_coil100,
    load_cifar100,
    load_weizmann,
]


def find_local_core_points_same(data, k, percent):
    n = data.shape[0]
    subset = int(np.floor(n * percent))

    p_dist = pairwise_distances(data, metric="euclidean")
    core_dists = np.partition(p_dist, k - 1, axis=0)[k - 1]

    nn = np.argpartition(p_dist, subset, axis=1)[:, :subset]
    # nn_mask = np.tile(np.arange(n).reshape(-1, 1), (1, subset))
    # nn = nn[nn != nn_mask].reshape(n, subset - 1)
    refined_medians = np.median(core_dists[nn], axis=1)

    vector_mask = core_dists < refined_medians
    return vector_mask, refined_medians


def find_local_core_points_adaptive(data, k, percent):
    n = data.shape[0]
    subset = int(np.floor(n * percent))

    p_dist = pairwise_distances(data, metric="euclidean")
    core_dists = np.partition(p_dist, k - 1, axis=0)[k - 1]

    nn = np.argpartition(p_dist, subset, axis=1)[:, :subset]
    # nn_mask = np.tile(np.arange(n).reshape(-1, 1), (1, subset))
    # nn = nn[nn != nn_mask].reshape(n, subset - 1)

    refined_medians = np.empty((n))
    for i in range(n):
        p_dist_neighbors = p_dist[np.ix_(nn[i], nn[i])]
        core_dists_neighbors = np.partition(p_dist_neighbors, k - 1, axis=0)[k - 1]
        refined_medians[i] = np.median(core_dists_neighbors)

    vector_mask = core_dists < refined_medians
    return vector_mask, refined_medians


def load_data_and_embedding(load_fn):
    data, gt_labels, data_name, _ = load_fn()

    PRETRAINED_MODELS_ROOT_PATH = (
        "/export/share/peters57dm/Verbund/deepsync/experiments/comparison102/ae_sync_loss/knn_label_assignment"
    )
    EXP_NO_NAME = "exp_00"
    BATCH_SIZE = 256
    MAX_EMBED_SIZE = 10
    embedded_space_dim = min(data.shape[1], MAX_EMBED_SIZE)

    model = Autoencoder(input_dim=data.shape[1], embedding_size=embedded_space_dim)
    trainloader, testloader = get_train_and_testloader(data, gt_labels, BATCH_SIZE)

    pretrained_model_path = os.path.join(
        PRETRAINED_MODELS_ROOT_PATH, data_name, EXP_NO_NAME, "pretrained_autoencoder.pth"
    )
    model = load_pretrained_model(model, pretrained_model_path, device="cpu")
    embedded, gt_labels = encode_batchwise(testloader, model, device="cpu")
    return data.numpy(), embedded, gt_labels, data_name


### Start Experiments ###
excludeTreeTypes = [
    UTreeType.LoadTree,
]
TREE_TYPES = [treeType for treeType in AVAILABLE_ULTRAMETRIC_TREE_TYPES if treeType not in excludeTreeTypes]
HIERACHIES = range(0, 5)
PARTITIONING_METHODS = AVAILABLE_PARTITIONING_METHODS

MIN_POINTS = 5
MIN_CLUSTER_SIZE = 15

for find_core_pts_fn, core_pts_name in [
    (find_local_core_points_same, "same_core_pts"),
    (find_local_core_points_adaptive, "adaptive_core_pts"),
]:
    for load_fn in datasets_loading_methods:
        original_data, embedded_data, gt_labels, data_name = load_data_and_embedding(load_fn)
        for data, space_name in [
            (original_data, "original_space"),
            (embedded_data, "embedding_space"),
        ]:
            savestring = f"./core_pts/{core_pts_name}##{data_name}##{space_name}.npy"
            if os.path.exists(savestring):
                print(f"Skipping: CORE_PTS: {core_pts_name}, DATASET: {data_name}, SPACE: {space_name}")
                continue

            print(f"Computing: CORE_PTS: {core_pts_name}, DATASET: {data_name}, SPACE: {space_name}")

            core_points_mask, _ = find_core_pts_fn(data, k=50, percent=0.1)

            os.makedirs(os.path.dirname(savestring), exist_ok=True)
            np.save(savestring, core_points_mask)
