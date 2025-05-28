#imports
import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from tqdm import tqdm
from helper import (
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

    get_local_core_points,
    load_data,
    get_train_and_testloader,
    detect_device,
    load_pretrained_model,
    encode_batchwise,
    save_dict_as_json,

    data_path,

    Autoencoder
)

from SHiP import SHiP
from SHiP.ultrametric_tree import UltrametricTreeType as UTreeType, AVAILABLE_ULTRAMETRIC_TREE_TYPES
from SHiP.partitioning import PartitioningMethod as PMethod, AVAILABLE_PARTITIONING_METHODS

# Parameters
max_embed_size = 10
batch_size = 256
pretrianed_models_root_path = "/export/share/peters57dm/Verbund/deepsync/experiments/comparison102/ae_sync_loss/knn_label_assignment"
exp_no_name = "exp_00"
k = 50
percent = 0.1
device = detect_device()

# SHiP Parameters
treeType = UTreeType.DCTree
power = 2
partitioningMethod = PMethod.ElbowOld
config = {}

experiment_path = "/export/share/peters57dm/Verbund/deepsync/experiments/SHiP_CorePoints_Clustering"
os.makedirs(experiment_path, exist_ok=True)

datasets_loading_methods = [load_pendigits,
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
                            load_weizmann,]

for loading_method in tqdm(datasets_loading_methods, total=len(datasets_loading_methods)):
    data, gt_labels, data_name = load_data(loading_method=loading_method)
    core_pts_path = os.path.join(data_path, 'corepts', f'{data_name}_corepts.npy')
    embedded_space_dim = min(data.shape[1], max_embed_size)
    model = Autoencoder(input_dim=data.shape[1], embedding_size=embedded_space_dim)
    trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
    pretrained_model_path = os.path.join(pretrianed_models_root_path, data_name, exp_no_name, 'pretrained_autoencoder.pth')
    model = load_pretrained_model(model, pretrained_model_path, device)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    core_points_mask, _ = get_local_core_points(embedded, k, percent, core_pts_path)

    # Core Points
    X = embedded[np.where(np.diag(core_points_mask)==1)[0]]
    true_labels = gt_labels[np.where(np.diag(core_points_mask)==1)[0]]

    # SHiP Clustering
    ship = SHiP(data=X, treeType=treeType, config=config)
    predicted_labels = ship.fit_predict(power, partitioningMethod)
    results = {
        "ARI" : ari(true_labels, predicted_labels),
        "AMI" : ami(true_labels, predicted_labels),
        "predictions" : predicted_labels
    }
    result_path = os.path.join(experiment_path, f"{data_name}_results.json")

    # Save the results
    save_dict_as_json(results, result_path)