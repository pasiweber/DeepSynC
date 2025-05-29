import os
import sys
import torch
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder

sys.path.append("/export/share/peters57dm/Verbund/deepsync/experiments/")
from helper.datasets import (
    # Data
    load_example,
    load_usps,
    load_htru,
    load_pendigits,
    load_optdigits,
    load_mnist,
    load_letterrecognition,
    load_fmnist,
    load_kmnist,
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
    load_data,
)
from experiments.helper.deep import (
    # other helper methods
    detect_device,
    get_train_and_testloader,
    encode_batchwise,
    load_pretrained_model,  #
)
from competitors.competitors import (
    # competitors initializers
    acedec_fit,
    dcn_fit,
    ddc_fit,
    dec_fit,
    dipdeck_fit,
    dkm_fit,
    idec_fit,
    hdbscan_fit,
    affinityprop_fit,
    meanshift_fit,
)
from helper.utils import save_dict_as_json
from helper.tracker import EvaluationTracker, update_eval_tracker


# region Exp. Definition
experiment_params = {
    "competitors": [
        # ("dcn", dcn_fit),
        # ("ddc", ddc_fit),
        # ("dipdeck", dipdeck_fit),
        # ("dkm", dkm_fit),
        # ("acedec", acedec_fit),
        # ("dec", dec_fit),
        # ("idec", idec_fit),
        # ("hdbscan", hdbscan_fit),
        # ("affinityprop", affinityprop_fit),
        ("meanshift", meanshift_fit),
    ],
    "datasets": [
        load_optdigits,
        load_letterrecognition,
        load_gaussian_blobs,
        load_example,
        load_pendigits,
        load_usps,
        load_htru,
        load_har,
        load_mice,
        load_synth_high,
        load_synth_low,
        load_mnist,
        load_fmnist,
        # load_kmnist,
        # load_cifar10,
        load_coil20,
        load_coil100,
        # load_cifar100,
        load_weizmann,
    ],
    "experiment_repetitions": 1,
}

# Hey! that is important too. Don't go too fast my friend :)
execution_params = {
    "experiment_root_path": "/export/share/peters57dm/Verbund/deepsync/results/experiments/competitors302",
    "model_name": "autoencoder.pth",
    "clustering_training_iterations": 150,
    "batch_size": 256,
    "max_embedded_dim_size": 10,
    "clustering_learning_rate": 1e-4,
    "do_pretrain": False,
    "pretrained_model_path": "/export/share/peters57dm/Verbund/data/n-pretrained-models/mse_pretrained_models256-128-64",
    "n_models": 5,
    "AE_layers": [256, 128, 64],
    "work_on_copy": False,
    "random_state": None,  # we repeat the experiments to get an estimation of the mean and std, no need for a random state
}
# endregion


# region Params Loading
base_path = execution_params["experiment_root_path"]
competitors = experiment_params["competitors"]
datasets_loading_methods = experiment_params["datasets"]
clustering_n_epochs = execution_params["clustering_training_iterations"]
batch_size = execution_params["batch_size"]
do_pretrain = execution_params["do_pretrain"]
experiment_repetitions = experiment_params["experiment_repetitions"]
max_embed_size = execution_params["max_embedded_dim_size"]
# device = "cuda:1"
device = detect_device()
model_name = execution_params["model_name"]
pretrained_model_name = "pretrained_" + model_name
pretrained_model_path = execution_params["pretrained_model_path"]
work_on_copy = execution_params["work_on_copy"]
random_state = execution_params["random_state"]
clustering_lr = execution_params["clustering_learning_rate"]
ae_layers = execution_params["AE_layers"]
N_MODELS = execution_params["n_models"]
# endregion

# Create Experiment Base Path
os.makedirs(base_path, exist_ok=True)

# Execution
n_experiments = len(competitors) * len(datasets_loading_methods) * experiment_repetitions
exp_follower = 1

# save the experiment settings
save_dict_as_json(execution_params, os.path.join(base_path, "experiment_settings.json"))

for competitor_name, comp_fit_method in competitors:
    competitor_path = os.path.join(base_path, competitor_name)
    os.makedirs(competitor_path, exist_ok=True)

    print(f"Competitor: {competitor_name}")
    for ds_loader in datasets_loading_methods:
        try:
            data, gt_labels, data_name = load_data(ds_loader)
            true_n_clusters = len(torch.unique(gt_labels))
            print(f"Dataset: {data_name}")
        except:  # handling any reason for this particular dataset loading failure
            print(f"Loading {data_name} failed. Continue to next experiment")
            continue
        dataset_experiment_path = os.path.join(competitor_path, data_name)
        os.makedirs(dataset_experiment_path, exist_ok=True)
        for exp_i in range(experiment_repetitions):
            for i in range(N_MODELS):

                exp_no_name = "exp_{0:02}".format(exp_i)
                experiment_directory_path = os.path.join(dataset_experiment_path, exp_no_name)
                os.makedirs(experiment_directory_path, exist_ok=True)
                print(f"Experiment {exp_follower} / {n_experiments}")
                exp_follower += 1
                embedded_space_dim = min(data.shape[1], max_embed_size)
                model = FeedforwardAutoencoder(
                    layers=[data.shape[1], ae_layers[0], ae_layers[1], ae_layers[2], embedded_space_dim]
                ).to(device)
                _mpath = os.path.join(pretrained_model_path, f"pretrained_{data_name}_{i}.pth")
                model = load_pretrained_model(model, _mpath, device)
                trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)

                # convert pytorch tensors to numpy
                np_data = data.numpy()
                np_gt_labels = gt_labels.numpy()
                eval_tracker = EvaluationTracker(dataset_size=len(np_data))

                try:
                    com = comp_fit_method(
                        n_clusters=true_n_clusters,
                        batch_size=batch_size,
                        device=device,
                        neural_network=model,
                        clustering_n_epochs=clustering_n_epochs,
                        clustering_optimizer_params={"lr": clustering_lr},
                        data=np_data,
                        random_state=random_state,
                    )
                except Exception as e:
                    save_dict_as_json(
                        {f"An exception raised while fitting {competitor_name}": str(e)},
                        os.path.join(experiment_directory_path, "error.json"),
                    )
                    continue

                eval_tracker = update_eval_tracker(np_gt_labels, com.labels_, eval_tracker)
                save_dict_as_json(eval_tracker.to_dict(), os.path.join(experiment_directory_path, f"results_{i}.json"))
                if hasattr(com, "neural_network_trained_"):
                    torch.save(
                        com.neural_network_trained_.state_dict(),
                        os.path.join(experiment_directory_path, "neural_network_trained_model.pth"),
                    )
