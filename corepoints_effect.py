import os
from tqdm import tqdm
import numpy as np
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.cluster import KMeans
from SHiP import SHiP
from helper import (
    load_data,

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
    load_coil20,
    load_coil100,
    load_weizmann,

    detect_device,
    save_dict_as_json,
    load_pretrained_model,
    find_local_core_points_same,

    encode_batchwise,
    get_train_and_testloader,
)

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
    load_coil20,
    load_coil100,
    load_weizmann,
]

device = detect_device()
experiments_path = "/export/share/peters57dm/Verbund/deepsync/experiments"
pretrained_models_path = "/export/share/peters57dm/Verbund/data/n-pretrained-models/mse_pretrained_models256-128-64"
experiment_name = "corepoints_effect"
batch_size = 256
max_embed_size = 10
ae_layers = [256, 128, 64]
k = 25
percent = 0.1
N_MODELS = 5

results = {}
for ds_loader in tqdm(datasets_loading_methods, total=len(datasets_loading_methods)):
    try :
            data, gt_labels, data_name = load_data(ds_loader)
            trainloader, testloader = get_train_and_testloader(data, gt_labels, batch_size)
            print(f"Dataset: {data_name}")
    except : # handling any reason for this particular dataset loading failure
        print(f"Loading {data_name} failed. Continue to next experiment")
        continue
    embedded_space_dim = min(data.shape[1], max_embed_size)
    gt_labels = gt_labels.numpy()
    for i in range(N_MODELS):   
        # initialize results dictionary
        results[f"{data_name}_{i}"] =  {}
        
        # load the data and model
        _mpath = os.path.join(pretrained_models_path, f"pretrained_{data_name}_{i}.pth")
        
        k_true = len(np.unique(gt_labels))
        model = FeedforwardAutoencoder(layers=[data.shape[1],
                                                ae_layers[0], ae_layers[1], ae_layers[2],
                                                embedded_space_dim]).to(device)
        model = load_pretrained_model(model, _mpath, device)
        embedded, gt_labels = encode_batchwise(testloader, model, device)
        core_points_mask, th = find_local_core_points_same(embedded, k, percent)
        core_points = embedded[np.where(np.diag(core_points_mask)==1)[0]]
        k_true_core = len(np.unique(gt_labels[np.where(np.diag(core_points_mask)==1)[0]]))

        # ship core
        ship_core = SHiP(data=core_points, treeType="DCTree")
        ship_core_labels = ship_core.fit_predict(power=2, partitioningMethod="ThreshholdElbow")
        results[f"{data_name}_{i}"]["ship_core_labels"] = ship_core_labels
        results[f"{data_name}_{i}"]["true_core_labels"] = gt_labels[np.where(np.diag(core_points_mask)==1)[0]].tolist()

        # ship embedded
        ship_embedded = SHiP(data=embedded, treeType="DCTree")
        ship_embedded_labels = ship_embedded.fit_predict(power=2, partitioningMethod="ThreshholdElbow")
        results[f"{data_name}_{i}"]["ship_embedded_labels"] = ship_embedded_labels
        results[f"{data_name}_{i}"]["true_embedded_labels"] = gt_labels.tolist()

        # kmeans core
        kmeans_core = KMeans(n_clusters=k_true_core, init="k-means++", n_init=10)
        kmeans_core_labels = kmeans_core.fit_predict(core_points)
        results[f"{data_name}_{i}"]["kmeans_core_labels"] = kmeans_core_labels.tolist()

        # kmeans embedded
        kmeans_embedded = KMeans(n_clusters=k_true, init="k-means++", n_init=10)
        kmeans_embedded_labels = kmeans_embedded.fit_predict(embedded)
        results[f"{data_name}_{i}"]["kmeans_embedded_labels"] = kmeans_embedded_labels.tolist()

os.makedirs(os.path.join(experiments_path, experiment_name), exist_ok=True)
save_dict_as_json(results, os.path.join(experiments_path, experiment_name, "kmeans_ship_labels.json"))