# region Imports
import os
import sys
import numpy as np
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from tqdm import tqdm
sys.path.append("/export/share/peters57dm/Verbund/deepsync/experiments/")

from helper.datasets import (
    load_example,
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

sys.path.append("/export/share/peters57dm/Verbund/deepsync/")
from helper.deep import (
    detect_device,
    load_pretrained_model,
    get_train_and_testloader,
    encode_batchwise
)
from helper.utils import save_dict_as_json, load_json_as_dict

# region Exp. Definition
experiment_params = {
    "datasets": [
        load_example,
        load_usps,
        load_htru,
        load_pendigits,
        load_optdigits,
        load_letterrecognition,
        load_har,
        load_mice,
        load_mnist,
        load_fmnist,
        load_coil20,
        load_coil100,
        load_weizmann,
    ],
}
# Hey! that is important too. Don't go too fast my friend :)
execution_params = {
    "experiment_root_path": "/export/share/peters57dm/Verbund/deepsync/results/experiments/HDBSCAN1NN",
    "batch_size": 256,
    "max_embedded_dim_size": 10,
    "pretrained_model_path": "/export/share/peters57dm/Verbund/data/n-pretrained-models/mse_pretrained_models256-128-64",
    "AE_layers": [256, 128, 64],
}
# endregion


# region Params Loading
base_path = execution_params["experiment_root_path"]
datasets_loading_methods = experiment_params["datasets"]

batch_size = execution_params["batch_size"]
max_embed_size = execution_params["max_embedded_dim_size"]

device = "cuda:1"
# device = detect_device()
pretrained_model_path = execution_params["pretrained_model_path"]
ae_layers = execution_params["AE_layers"]
N_MODELS = 5
# endregion

def load_hdbscan_predictions(dataname, model_i):
    results_path = f"/export/share/peters57dm/Verbund/deepsync/results/experiments/competitors302/hdbscan/{dataname}/exp_00/results_{model_i}.json"
    results = load_json_as_dict(results_path)
    return np.array(results["predicted_labels"])

results = {}
for ds_loader in tqdm(datasets_loading_methods, total=len(datasets_loading_methods)):
    data, gt_labels, data_name = load_data(ds_loader)
    embedded_space_dim = min(data.shape[1], max_embed_size)
    for model_i in range(N_MODELS):
        
        model = FeedforwardAutoencoder(
            layers=[data.shape[1], ae_layers[0], ae_layers[1], ae_layers[2], embedded_space_dim]
        ).to(device)
        _mpath = os.path.join(pretrained_model_path, f"pretrained_{data_name}_{model_i}.pth")
        model = load_pretrained_model(model, _mpath, device)
        _, testloader = get_train_and_testloader(data, gt_labels, batch_size)
        embedded, gt_labels = encode_batchwise(testloader, model, device)
        np_gt_labels = gt_labels.numpy()

        predicted_labels = load_hdbscan_predictions(data_name, model_i)
        labeled_points_mask = predicted_labels > -1
        labeled_points = embedded[labeled_points_mask]
        corresponding_labels = np_gt_labels[labeled_points_mask]
        if len(embedded[~labeled_points_mask]) == 0:
            if not data_name in results:
                results[data_name] = {}
            results[data_name][f"model_0{model_i}"] = "All points are alread labeled."
            continue
        knn = KNN(1).fit(labeled_points, corresponding_labels)
        knn_predictions = knn.predict(embedded[~labeled_points_mask])
        nn_predictions = np.copy(np_gt_labels)
        nn_predictions[~labeled_points_mask] = knn_predictions
        if not data_name in results:
            results[data_name] = {}
        results[data_name][f"model_0{model_i}"] = {}
        results[data_name][f"model_0{model_i}"]["NN_predictions"] = nn_predictions.tolist()
        results[data_name][f"model_0{model_i}"]["original_predictions"] = predicted_labels.tolist()
        results[data_name][f"model_0{model_i}"]["gt_labels"] = np_gt_labels.tolist()
        results[data_name][f"model_0{model_i}"]["ami"] = ami(nn_predictions, predicted_labels)
        results[data_name][f"model_0{model_i}"]["ari"] = ari(nn_predictions, predicted_labels)
save_dict_as_json(
    results, os.path.join(base_path, f"pretrained_models_results.json")
)