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
)
from DeepSynC.ablation_version import deep_sync_model_ship
from helper.deep import (
    detect_device,
    load_pretrained_model,
    get_train_and_testloader,
)

import traceback
from itertools import product


# default setting: k = 25, percent = 0.1, n_check = 3, T_stop stopping

# -) k nearest neigbour and for majority vote [5, 15, 25,35, 45]
# -) percent [0.05, 0.075, 0.1, 0.125, 0.15] --> check number of core points, do they capture the gt clusters
# -) n_check in [2,3,4,5]
# -) T_stop for early stopping [3,5,7,9]

NR_MODELS = 5
parameter_combinations = [
    {
        "k": [5, 15, 25, 35, 45],
        "percent": [0.1],
        "n_check": [3],
        "T_stop": [3],
    },
    {
        "k": [25],
        "percent": [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
        "n_check": [3],
        "T_stop": [3],
    },
    {
        "k": [25],
        "percent": [0.1],
        "n_check": [2, 3, 4, 5, 6, 7],
        "T_stop": [3],
    },
    {
        "k": [25],
        "percent": [0.1],
        "n_check": [3],
        "T_stop": [3, 5, 7, 9, 11, 13],
    },
]


def run(k=25, percent=0.1, n_check=3, T_stop=3, id=0, model_id=0):
    print("")
    try:
        device = "cuda"  # detect_device()
        deepsync_kwargs = {
            "device": device,
            "training_iterations": 300,
            "k": k,
            "percent": percent,
            "n_check": n_check,
            "T_stop": T_stop,
            "id": id,
            "use_tqdm": True,
        }

        experiment_params = {
            "loss_funs": [
                ("ae_sync_loss", ae_sync_loss),
                # ("att_rep_loss", attraction_repelling_loss),
            ],  # number of loss functions must be equal to # number of losses trackers
            "batch_loss_trackers": [
                AESyncLossTracker,
                # AttractionRepellingLossTracker,
            ],
            "evaluation_tracker": EvaluationTracker,
            "datasets": [
                # load_example,
                # load_mice,
                # load_optdigits,
                # load_har,
                # load_usps,
                load_htru,
                load_pendigits,
                load_letterrecognition,
                load_coil20,
                load_mnist,
            ],
            "label_assignment_methods": [
                ("knn_label_assignment", knn_assign_unlabeled_points),
                # ("mahalanobis_label_assignment", mahalanobis_assign_unlabeled_points),
                # ("knn_average_dist_label_assignment", knn_average_assign_unlabeled_points),
                # ("vote_knn_methods", vote_of_two_knn_methods),
                # ("euclidean_label_assignment", euclidean_assign_unlabeled_points),
            ],
        }
        assert len(experiment_params["loss_funs"]) == len(experiment_params["batch_loss_trackers"])

        batch_size = 256
        base_path = (
            f"/export/share/peters57dm/Verbund/deepsync/results/ablations/{k=}##{percent=}##{n_check=}##{T_stop=}/"
        )

        model_name = "autoencoder.pth"
        max_embed_size = 10
        ae_layers = [256, 128, 64]
        learning_rate_deepsync = 1e-4

        pretrained_model_path = (
            "/export/share/peters57dm/Verbund/data/n-pretrained-models/mse_pretrained_models256-128-64"
        )
        deep_sync_model_name = "deep_sync_" + model_name
        deep_sync_path = os.path.join(base_path, "deep_sync.pth")

        loss_fns = experiment_params["loss_funs"]
        datasets_loading_methods = experiment_params["datasets"]
        loss_trackers = experiment_params["batch_loss_trackers"]
        labels_assignment_methods = experiment_params["label_assignment_methods"]

        # Create Experiment Base Path
        os.makedirs(base_path, exist_ok=True)

        for lf, LossTracker in zip(loss_fns, loss_trackers):
            lf_name, loss_fun_method = lf
            lf_path = os.path.join(base_path, lf_name)
            os.makedirs(lf_path, exist_ok=True)

            for lam in labels_assignment_methods:
                label_assignment_method_name, label_assignment_method = lam
                lam_path = os.path.join(lf_path, label_assignment_method_name)
                os.makedirs(lam_path, exist_ok=True)

                print(f"Loss: {lf_name} - Label Assignment: {label_assignment_method_name} - Device: {device}")
                for ds_loader in datasets_loading_methods:
                    try:
                        data, gt_labels, data_name = load_data(ds_loader)
                        trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
                        # print(f"\n\nDataset: {data_name}")
                    except:  # handling any reason for this particular dataset loading failure
                        print(f"Loading {data_name} failed. Continue to next experiment")
                        continue
                    dataset_experiment_path = os.path.join(lam_path, data_name)
                    os.makedirs(dataset_experiment_path, exist_ok=True)

                    # experiment_directory_path = os.path.join(dataset_experiment_path, 'exp_{0:02}'.format(exp_i)) # because we don't do experiment repetitions, but we use different pretrained models.
                    experiment_directory_path = os.path.join(dataset_experiment_path, "model_{0:02}".format(model_id))
                    os.makedirs(experiment_directory_path, exist_ok=True)

                    eval_tracker_path = f"/export/share/peters57dm/Verbund/deepsync/results/ablations/k={k}##percent={percent}##n_check={n_check}##T_stop={T_stop}/{lf_name}/{label_assignment_method_name}/{data_name}/{"model_{0:02}".format(model_id)}/trackers/eval_tracker.json"
                    if os.path.exists(eval_tracker_path):
                        print(
                            f"{id+1}/{NR_COMB}: k={k}##percent={percent}##n_check={n_check}##T_stop={T_stop}/{lf_name}/{label_assignment_method_name}/{data_name}/{"model_{0:02}".format(model_id)} -- Already exists"
                        )
                        continue

                    print(
                        f"{id+1}/{NR_COMB}: k={k}##percent={percent}##n_check={n_check}##T_stop={T_stop}/{lf_name}/{label_assignment_method_name}/{data_name}/{"model_{0:02}".format(model_id)} -- Compute..."
                    )

                    embedded_space_dim = min(data.shape[1], max_embed_size)
                    model = FeedforwardAutoencoder(
                        layers=[data.shape[1], ae_layers[0], ae_layers[1], ae_layers[2], embedded_space_dim]
                    ).to(device)
                    _mpath = os.path.join(pretrained_model_path, f"pretrained_{data_name}_{model_id}.pth")
                    model = load_pretrained_model(model, _mpath, device)

                    deepsync_kwargs["model"] = model
                    deepsync_kwargs["data"] = data
                    deepsync_kwargs["dataset_name"] = data_name
                    deepsync_kwargs["trainloader"], deepsync_kwargs["testloader"] = get_train_and_testloader(
                        data, gt_labels, batch_size
                    )
                    deepsync_kwargs["optimizer"] = torch.optim.Adam(model.parameters(), lr=learning_rate_deepsync)
                    deepsync_kwargs["loss_fn"] = loss_fun_method
                    deepsync_kwargs["labels_assignment_method"] = label_assignment_method
                    deepsync_kwargs["losses_tracker"] = LossTracker()
                    deepsync_kwargs["eval_tracker"] = EvaluationTracker(dataset_size=len(data))
                    deepsync_kwargs["directory"] = experiment_directory_path
                    deepsync_kwargs["deep_sync_model_path"] = os.path.join(
                        experiment_directory_path, deep_sync_model_name
                    )

                    model, labels_over_iterations = deep_sync_model_ship(**deepsync_kwargs)

    except Exception as e:
        print(f"ERROR -- {k=}, {percent=}, {n_check=}, {T_stop=}")
        print(f"Exception: {e}")
        traceback.print_exc()
        return


unique_combinations = {
    tuple(zip(comb.keys(), values)) for comb in parameter_combinations for values in product(*comb.values())
}
unique_combinations = [dict(comb) for comb in unique_combinations]
unique_combinations = [dict({"model_id": model_id}, **d) for model_id in range(NR_MODELS) for d in unique_combinations]
unique_combinations = [dict({"id": id}, **comb) for (id, comb) in enumerate(unique_combinations)]

NR_COMB = len(unique_combinations)
print(NR_COMB)


for comb in unique_combinations:
    run(**comb)
