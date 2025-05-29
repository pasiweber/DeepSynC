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
        # load_example,
        # load_usps,
        # load_htru,
        # load_pendigits,
        # load_optdigits,
        # load_letterrecognition,
        load_har,
        load_mice,
        load_mnist,
        load_fmnist,
        load_coil20,
        load_coil100,
        load_weizmann,
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
    "experiment_root_path": "/export/share/peters57dm/Verbund/deepsync/results/experiments/comparison405",
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
    "Note": "In this experiment, we use SHiP with ground truth number of labels",
    "Note2": "Starting from code 400 and above, we used the new case of labeling assignment methods where only high confidence points can say something",
    "Note3": "removing the unified label method, 404 and 405 should be complementary to each other, where we use the best combination of deepsync and run for 5 pretrained models",
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
device = "cuda:1"
# device = detect_device()
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
                    if do_deepsync_train:
                        deep_sync_model_path = os.path.join(experiment_directory_path, deep_sync_model_name)
                        loss_tracker = LossTracker()
                        eval_tracker = EvaluationTracker(dataset_size=len(data))
                        model, labels_over_iterations = deep_sync_model_ship_trueK(
                            device,
                            model,
                            data,
                            data_name,
                            trainloader,
                            testloader,
                            optimizer,
                            n_check,
                            k,
                            percent,
                            clustering_n_epochs,
                            loss_fun_method,
                            label_assignment_method,
                            loss_tracker,
                            eval_tracker,
                            experiment_directory_path,
                            deep_sync_model_path,
                        )
