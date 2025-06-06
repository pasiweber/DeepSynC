from deepsync import fit_deepsync

# region Imports
import os
import torch
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder

from helper.tracker import (
    AESyncLossTracker,
    EvaluationTracker,
)
from helper.datasets import (
    load_usps,
    load_htru,
    load_pendigits,
    load_optdigits,
    load_mnist,
    load_letterrecognition,
    load_coil20,
    load_coil100,
    load_har,
    load_mice,
    load_weizmann,
    load_fmnist,
    load_data,
)

from helper.deepsync_utils import (
    # label assignment
    knn_assign_unlabeled_points,

    # loss function
    ae_sync_loss,
)
from helper.deep import (
    detect_device,
    pretrain_model,
    get_train_and_testloader,
)
# endregion

# region Params Loading
base_path = "./experiment"
datasets_loading_methods = [
    load_usps,
    load_htru,
    load_pendigits,
    load_optdigits,
    load_mnist,
    load_letterrecognition,
    load_coil20,
    load_coil100,
    load_har,
    load_mice,
    load_weizmann,
    load_fmnist,
]

# parameters definition
k = 25
percent = 0.1
n_check = 3
t_stop = 3
pretrain_lr = 1e-3
deepsync_lr = 1e-4
pretrain_n_epochs = 100
clustering_n_epochs = 300
batch_size = 256
max_embed_size = 10
do_pretrain = True
do_deepsync_train = True
pretrain_loss_fn = torch.nn.MSELoss()
device = detect_device()
model_name = "ae_mlp"
pretrained_model_path = f"./pretrained_models"
deep_sync_model_name = "deep_sync_" + model_name + ".pth"
deep_sync_path = os.path.join(base_path, "deep_sync.pth")
ae_layers = [256, 128, 64]
N_MODELS = 5
# endregion

def main():
    os.makedirs(pretrained_model_path, exist_ok=True)
    os.makedirs(base_path, exist_ok=True)
    for ds_loader in datasets_loading_methods:
        try:
            data, gt_labels, data_name = load_data(ds_loader)
            trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
            print(f"Dataset: {data_name}")
        except:  # handling any reason for this particular dataset loading failure
            print(f"Loading {data_name} failed. Continue to next experiment")
            continue
        dataset_experiment_path = os.path.join(base_path, data_name)
        os.makedirs(dataset_experiment_path, exist_ok=True)
        for i in range(N_MODELS):
            experiment_directory_path = os.path.join(dataset_experiment_path, "model_{0:02}".format(i))
            os.makedirs(experiment_directory_path, exist_ok=True)

            embedded_space_dim = min(data.shape[1], max_embed_size)

            model = FeedforwardAutoencoder(
                layers=[data.shape[1], ae_layers[0], ae_layers[1], ae_layers[2], embedded_space_dim]
            ).to(device)
            pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
            model = pretrain_model(
                        device,
                        os.path.join(pretrained_model_path, f"{model_name}_{data_name}_i.pth"),
                        model,
                        data,
                        trainloader,
                        pretrain_optimizer,
                        pretrain_loss_fn,
                        training_iterations=pretrain_n_epochs,
                        verbose=True,
                    )
            optimizer = torch.optim.Adam(model.parameters(), lr=deepsync_lr)
            trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
            if do_deepsync_train:
                deep_sync_model_path = os.path.join(experiment_directory_path, deep_sync_model_name)
                loss_tracker = AESyncLossTracker()
                eval_tracker = EvaluationTracker(dataset_size=len(data))
                model, labels_over_iterations = fit_deepsync(
                    device,
                    model,
                    data,
                    data_name,
                    trainloader,
                    testloader,
                    optimizer,
                    n_check,
                    t_stop,
                    k,
                    percent,
                    clustering_n_epochs,
                    ae_sync_loss,
                    knn_assign_unlabeled_points,
                    loss_tracker,
                    eval_tracker,
                    experiment_directory_path,
                    deep_sync_model_path,
                    id=0,
                    use_tqdm=False,
                )


if __name__ == "__main__":
    main()