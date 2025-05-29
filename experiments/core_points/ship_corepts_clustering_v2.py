import os
import sys
import numpy as np
import glob
from mpire.pool import WorkerPool

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
from .ship_corepts_clustering_v2_core_pts import find_local_core_points_same, find_local_core_points_adaptive


datasets_loading_methods = [
    (load_pendigits, "pendigits"),
    (load_optdigits, "optdigits"),
    (load_letterrecognition, "letterrecognition"),
    (load_gaussian_blobs, "easy_blobs"),
    (load_example, "example"),
    (load_usps, "USPS"),
    (load_htru, "htru"),
    (load_har, "HAR"),
    (load_mice, "mice"),
    (load_synth_high, "synth_high"),
    (load_synth_low, "synth_low"),
    (load_mnist, "MNIST"),
    (load_fmnist, "FMNIST"),
    (load_cifar10, "cifar10"),
    (load_coil20, "coil20"),
    (load_coil100, "coil100"),
    (load_cifar100, "cifar100"),
    (load_weizmann, "weizmann"),
]


def load_data_and_embedding(load_fn, data_name):
    savestring_data = f"./.cache/{data_name}##data.npy"
    savestring_embedding = f"./.cache/{data_name}##embedding.npy"
    savestring_gt_labels = f"./.cache/{data_name}##gt_labels.npy"

    if (
        os.path.exists(savestring_data)
        and os.path.exists(savestring_embedding)
        and os.path.exists(savestring_gt_labels)
    ):
        return np.load(savestring_data), np.load(savestring_embedding), np.load(savestring_gt_labels)

    data, gt_labels, _, _ = load_fn()

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

    os.makedirs(os.path.dirname(savestring_data), exist_ok=True)
    np.save(savestring_data, data)
    np.save(savestring_embedding, embedded)
    np.save(savestring_gt_labels, gt_labels)

    return data, embedded, gt_labels


### Start Experiments ###
excludeTreeTypes = [
    UTreeType.LoadTree,
]
TREE_TYPES = [treeType for treeType in AVAILABLE_ULTRAMETRIC_TREE_TYPES if treeType not in excludeTreeTypes]
HIERACHIES = range(0, 5)
PARTITIONING_METHODS = AVAILABLE_PARTITIONING_METHODS

MIN_POINTS = 5
MIN_CLUSTER_SIZE = 15


def get_core_pts(core_pts_name, data_name, space_name, find_core_pts_fn, data):
    savestring = f"./core_pts/{core_pts_name}##{data_name}##{space_name}.npy"
    if os.path.exists(savestring):
        print(f"Loading: CORE_PTS: {core_pts_name}, DATASET: {data_name}, SPACE: {space_name}")
        return np.load(savestring)

    print(f"Computing: CORE_PTS: {core_pts_name}, DATASET: {data_name}, SPACE: {space_name}")

    core_points_mask, _ = find_core_pts_fn(data, k=50, percent=0.1)

    os.makedirs(os.path.dirname(savestring), exist_ok=True)
    np.save(savestring, core_points_mask)
    return core_points_mask


def run_ship(core_pts_name, data_name, space_name, find_core_pts_fn, data, gt_labels, treeType):
    core_points_mask = get_core_pts(core_pts_name, data_name, space_name, find_core_pts_fn, data)
    core_points = data[core_points_mask]
    gt_labels = gt_labels[core_points_mask]

    k = len(np.unique(gt_labels))

    print(
        f"Running: CORE_PTS: {core_pts_name}, DATASET: {data_name}, SPACE: {space_name}, treeType: {treeType}, n: {len(core_points)}, dim: {len(core_points[0])}, k: {k}"
    )
    config = {
        "k": k,
        "min_points": MIN_POINTS,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "optimize_tree": True,
    }
    ship = SHiP(data=core_points, treeType=treeType, config=config)

    for power in HIERACHIES:
        for partitioningMethod in PARTITIONING_METHODS:
            ship.power = power
            ship.partitioningMethod = partitioningMethod

            labels = ship.fit_predict(power, partitioningMethod)

            savestring = (
                f"./labels/{core_pts_name}##{data_name}##{space_name}##{treeType}##{power}##{partitioningMethod}.npy"
            )
            os.makedirs(os.path.dirname(savestring), exist_ok=True)
            np.save(savestring, labels)


pool = WorkerPool(n_jobs=20, use_dill=True)

for treeType in TREE_TYPES:
    for find_core_pts_fn, core_pts_name in [
        (find_local_core_points_same, "same_core_pts"),
        (find_local_core_points_adaptive, "adaptive_core_pts"),
    ]:
        for load_fn, data_name in datasets_loading_methods:
            original_data, embedded_data, gt_labels = load_data_and_embedding(load_fn, data_name)
            for data, space_name in [
                (original_data, "original_space"),
                (embedded_data, "embedding_space"),
            ]:
                savestring = f"./labels/{core_pts_name}##{data_name}##{space_name}##{treeType}##"
                if len(glob.glob(savestring + "*")) == len(HIERACHIES) * len(PARTITIONING_METHODS):
                    print(
                        f"Skipping: CORE_PTS: {core_pts_name}, DATASET: {data_name}, SPACE: {space_name}, treeType: {treeType}"
                    )
                    continue

                pool.apply_async(
                    run_ship, args=(core_pts_name, data_name, space_name, find_core_pts_fn, data, gt_labels, treeType)
                )

pool.stop_and_join()
pool.terminate()
