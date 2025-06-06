import os
import pickle
import torch
from clustpy.deep.enrc import ACeDeC
from clustpy.deep.dcn import DCN
from clustpy.deep.ddc_n2d import DDC
from clustpy.deep.dec import DEC
from clustpy.deep.dipdeck import DipDECK
from clustpy.deep.dkm import DKM
from clustpy.deep.dec import IDEC
from sklearn.cluster import KMeans, HDBSCAN, AffinityPropagation, MeanShift

import sys
sys.path.append("/export/share/peters57dm/Verbund/deepsync/experiments/")
sys.path.append("/export/share/peters57dm/Verbund/deepsync/")
from helper.sync_raw import SyncClus

def kmeans_clustering(data, n_clusters, base_path, data_name=None, normalize=None, input_centers=None, random_state=7):
    n_init = 1
    if input_centers is None:
        input_centers = "k-means++"
        n_init = 10
    kmeans = KMeans(n_clusters=n_clusters, init=input_centers, n_init=n_init, random_state=random_state)
    kmeans_file_name = "{0}/kmeans_{1}_{2}_norm_{3}.pkl".format(base_path, data_name, n_clusters, normalize)
    if os.path.exists(kmeans_file_name):
        print("loading kmeans")
        kmeans = pickle.load(open(kmeans_file_name, "rb"))
    else:
        print("executing kmeans")
        kmeans.fit(data)
        if data_name is not None:
            print("saving kmeans")
            pickle.dump(kmeans, open(kmeans_file_name, "wb"))
    return kmeans


def acedec_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return ACeDeC(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state,
    ).fit(data)


def dcn_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return DCN(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state,
    ).fit(data)


def ddc_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return DDC(
        batch_size=batch_size,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state,
    ).fit(data)


def dec_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return DEC(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state,
    ).fit(data)


def dipdeck_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return DipDECK(
        n_clusters_init=int(n_clusters * 3),
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        # clustering_epochs=clustering_n_epochs, default is 50.
        device=device,
        random_state=random_state,
    ).fit(data)


def dkm_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return DKM(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state,
    ).fit(data)


def idec_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    return IDEC(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state,
    ).fit(data)


def _embed_data_np(data, batch_size, neural_network):
    from clustpy.deep import encode_batchwise as encode
    from clustpy.deep import get_dataloader as _dl

    dl = _dl(data, batch_size, shuffle=False)
    embedded = encode(dl, neural_network)

    return embedded


def hdbscan_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    embedded = _embed_data_np(data, batch_size, neural_network)
    return HDBSCAN().fit(embedded)


def affinityprop_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    embedded = _embed_data_np(data, batch_size, neural_network)
    return AffinityPropagation(random_state=random_state).fit(embedded)

def meanshift_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    embedded = _embed_data_np(data, batch_size, neural_network)
    return MeanShift().fit(embedded)

def sync_fit(
    n_clusters,
    batch_size,
    device,
    neural_network,
    clustering_n_epochs,
    clustering_optimizer_params,
    data,
    random_state,
):
    embedded = _embed_data_np(data, batch_size, neural_network)
    embedded = torch.from_numpy(embedded)
    sync = SyncClus()
    labels = sync.fit_predict(embedded)
    return sync
