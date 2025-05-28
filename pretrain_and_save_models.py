#imports
import os
import torch
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

    load_data,
    detect_device,
    save_dict_as_json
)
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from clustpy.deep import get_dataloader

# Parameters
max_embed_size = 10
batch_size = 256
n_epochs = 100 # 50
lr = 1e-3
ae_layers = [256, 128, 64]
N_MODELS = 5
pretrianed_models_path = f"/export/share/peters57dm/Verbund/data/n-pretrained-models/mse_pretrained_models{ae_layers[0]}-{ae_layers[1]}-{ae_layers[2]}"
device = detect_device()

os.makedirs(pretrianed_models_path, exist_ok=True)

datasets_loading_methods = [
                            load_example,
                            load_pendigits,
                            load_optdigits,
                            load_letterrecognition,
                            load_gaussian_blobs,
                            load_usps,
                            load_htru,
                            load_har,
                            load_mice,
                            load_synth_high,
                            load_synth_low,
                            load_mnist,
                            load_fmnist,
                            load_cifar10,
                            load_cifar100,
                            load_coil20,
                            load_coil100,
                            load_weizmann,
                            ]
for loading_method in tqdm(datasets_loading_methods, total=len(datasets_loading_methods)):
    for i in range(N_MODELS):
        ssl_loss_fn = torch.nn.MSELoss()
        data, gt_labels, data_name = load_data(loading_method=loading_method)
        embedded_space_dim = min(data.shape[1], max_embed_size)
        model = FeedforwardAutoencoder(layers=[data.shape[1],
                                                ae_layers[0], ae_layers[1], ae_layers[2],
                                                embedded_space_dim]).to(device)
        trainloader = get_dataloader(data, batch_size, shuffle=True)
        _pretrained_model_path = os.path.join(pretrianed_models_path, f'pretrained_{data_name}_{i}.pth')
        model.fit(n_epochs=n_epochs, optimizer_params={"lr":lr},
                  ssl_loss_fn=ssl_loss_fn, dataloader=trainloader, model_path=_pretrained_model_path,
                optimizer_class=lambda params, lr: torch.optim.AdamW(params, lr))
        mse_loss = model.evaluate(trainloader, ssl_loss_fn, device)
        _res = {f'pretrained_{data_name}_{i}' : float(mse_loss)}
        save_dict_as_json(_res, os.path.join(pretrianed_models_path, f'pretrained_{data_name}_{i}_loss.json'))