# region Imports
import os
import sys
import torch
import numpy as np
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder

sys.path.append("/export/share/peters57dm/Verbund/deepsync/experiments/")
from helper.tracker import (
    AttractionRepellingLossTracker,
    AESyncLossTracker,
    EvaluationTracker,
)
from helper.datasets import (
    load_example,
    load_usps,
    load_htru,
    load_pendigits,
    load_optdigits,
    load_mnist,
    load_letterrecognition,
    load_cmu_faces,
    load_coil20,
    load_coil100,
    load_har,
    load_mice,
    load_synth_high,
    load_synth_low,
    load_weizmann,
    load_gaussian_blobs,
    load_cifar10,
    load_cifar100,
    load_fmnist,
    load_data,
)

sys.path.append("/export/share/peters57dm/Verbund/deepsync/")
from DeepSynC.helper import (
    # label assignment
    knn_assign_unlabeled_points,
    mahalanobis_assign_unlabeled_points,
    euclidean_assign_unlabeled_points,
    knn_average_assign_unlabeled_points,
    vote_of_two_knn_methods,
    # loss functions
    attraction_repelling_loss,
    ae_sync_loss,
    # run deep sync
    deep_sync_model_ship_trueK,
    deep_sync_model_ship,
    encode_batchwise,
    find_local_core_points_same
)
from helper.deep import (
    detect_device,
    load_pretrained_model,
    get_train_and_testloader,
)
from helper.utils import save_dict_as_json

# endregion


# region Exp. Definition
experiment_params = {
    "loss_funs": [
        ("ae_sync_loss", ae_sync_loss),
        # ("att_rep_loss", attraction_repelling_loss)
    ],  # number of loss functions must be equal to # number of losses trackers
    "batch_loss_trackers": [
        AESyncLossTracker,
        # AttractionRepellingLossTracker
    ],
    "evaluation_tracker": EvaluationTracker,
    "datasets": [
        #load_example,
        #load_usps,
        #load_htru,
        #load_pendigits,
        #load_optdigits,
        #load_letterrecognition,
        #load_har,
        #load_mice,
        load_mnist,
        #load_fmnist,
        #load_coil20,
        #load_coil100,
        #load_weizmann,
        # load_synth_high,
        # load_synth_low,
        # load_cifar100,
        # load_cifar10,
        # load_gaussian_blobs,
    ],
    "label_assignment_methods": [
        ("knn_label_assignment", knn_assign_unlabeled_points),
        # ("mahalanobis_label_assignment", mahalanobis_assign_unlabeled_points),
        # ("knn_average_dist_label_assignment", knn_average_assign_unlabeled_points),
        # ("vote_knn_methods", vote_of_two_knn_methods)
        # ("euclidean_label_assignment", euclidean_assign_unlabeled_points)
    ],
    "experiment_repetitions": 1,
}
assert len(experiment_params["loss_funs"]) == len(experiment_params["batch_loss_trackers"])
# Hey! that is important too. Don't go too fast my friend :)
execution_params = {
    "experiment_root_path": "/export/share/peters57dm/Verbund/deepsync/results/experiments/core_point_cluster_coverage",
    "model_name": "autoencoder.pth",
    "k": 25,
    "percent": 0.1,
    "n_check": 3,  # check for high confidence and check for early stopping.
    "learning_rate_pretrain": None,
    "learning_rate_deepsync": 1e-4,
    "pretrain_training_iterations": None,
    "clustering_training_iterations": 300,  # Switched to a max iteration of 300.
    "batch_size": 256,
    "max_embedded_dim_size": 10,
    "do_pretrain": False,
    "pretrain_loss": torch.nn.MSELoss(),
    "do_deepsync_train": True,
    "gif_duration": 300,
    "pretrained_model_path": "/export/share/peters57dm/Verbund/data/n-pretrained-models/mse_pretrained_models256-128-64",
    "AE_layers": [256, 128, 64],
    "Note": "just extracting the core point coverage",
}
# endregion


# region Params Loading
base_path = execution_params["experiment_root_path"]
loss_fns = experiment_params["loss_funs"]
datasets_loading_methods = experiment_params["datasets"]
loss_trackers = experiment_params["batch_loss_trackers"]
labels_assignment_methods = experiment_params["label_assignment_methods"]
k = execution_params["k"]
percent = execution_params["percent"]
n_check = execution_params["n_check"]
pretrain_lr = execution_params["learning_rate_pretrain"]
deepsync_lr = execution_params["learning_rate_deepsync"]
pretrain_n_epochs = execution_params["pretrain_training_iterations"]
clustering_n_epochs = execution_params["clustering_training_iterations"]
batch_size = execution_params["batch_size"]
max_embed_size = execution_params["max_embedded_dim_size"]
do_pretrain = execution_params["do_pretrain"]
do_deepsync_train = execution_params["do_deepsync_train"]
pretrain_loss_fn = execution_params["pretrain_loss"]
experiment_repetitions = experiment_params["experiment_repetitions"]
# device = "cuda:1"
device = detect_device()
model_name = execution_params["model_name"]
pretrained_model_path = execution_params["pretrained_model_path"]
deep_sync_model_name = "deep_sync_" + model_name
deep_sync_path = os.path.join(base_path, "deep_sync.pth")
ae_layers = execution_params["AE_layers"]
N_MODELS = 5
# endregion


# Create Experiment Base Path
os.makedirs(base_path, exist_ok=True)

# save_dict_as_json(execution_params,
#             os.path.join(base_path, "experiment_settings.json"))#Execution

n_experiments = (
    len(loss_fns) * len(labels_assignment_methods) * len(datasets_loading_methods) * experiment_repetitions * N_MODELS
)
exp_follower = 1
for lf, LossTracker in zip(loss_fns, loss_trackers):
    lf_name, loss_fun_method = lf
    lf_path = os.path.join(base_path, lf_name)
    os.makedirs(lf_path, exist_ok=True)

    for lam in labels_assignment_methods:
        label_assignment_method_name, label_assignment_method = lam
        lam_path = os.path.join(lf_path, label_assignment_method_name)
        os.makedirs(lam_path, exist_ok=True)

        print(f"Loss: {lf_name} - Label Assignment: {label_assignment_method_name}")
        for ds_loader in datasets_loading_methods:
            try:
                data, gt_labels, data_name = load_data(ds_loader)
                trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
                print(f"Dataset: {data_name}")
            except:  # handling any reason for this particular dataset loading failure
                print(f"Loading {data_name} failed. Continue to next experiment")
                continue
            dataset_experiment_path = os.path.join(lam_path, data_name)
            os.makedirs(dataset_experiment_path, exist_ok=True)
            for exp_i in range(experiment_repetitions):
                for i in range(N_MODELS):
                    # experiment_directory_path = os.path.join(dataset_experiment_path, 'exp_{0:02}'.format(exp_i)) # because we don't do experiment repetitions, but we use different pretrained models.
                    experiment_directory_path = os.path.join(dataset_experiment_path, "model_{0:02}".format(i))
                    os.makedirs(experiment_directory_path, exist_ok=True)
                    print(f"Experiment {exp_follower} / {n_experiments}")
                    exp_follower += 1
                    embedded_space_dim = min(data.shape[1], max_embed_size)
                    model = FeedforwardAutoencoder(
                        layers=[data.shape[1], ae_layers[0], ae_layers[1], ae_layers[2], embedded_space_dim]
                    ).to(device)
                    _mpath = os.path.join(pretrained_model_path, f"pretrained_{data_name}_{i}.pth")
                    model = load_pretrained_model(model, _mpath, device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=deepsync_lr)
                    trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
                    
                    embedded_data, embedded_labels = encode_batchwise(testloader, model, device)
                    mask_same_comment, th_same_comment = find_local_core_points_same(embedded_data, k, percent)
                    core_points = embedded_data[np.where(np.diag(mask_same_comment) == 1)[0]]
                    core_point_labels = embedded_labels[np.where(np.diag(mask_same_comment) == 1)[0]]
                    gt_uniques = np.unique(core_point_labels, return_counts=True)
                    n_clusters = len(np.unique(core_point_labels))
                    print("ration_core_points:", core_points.shape[0]/embedded_data.shape[0])
                    print("Class distribution:\n", dict(zip(gt_uniques[0], gt_uniques[1])))

