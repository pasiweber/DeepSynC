import os
import torch
import numpy as np
from clustpy.data.preprocessing import ZNormalizer


data_path = "./data/benchmark"

def _convert_data_labels_to_torch(data, labels):
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()
    return data, labels


def load_har():
    from clustpy.data import load_har as har
    data, labels = har(return_X_y=True, downloads_path=data_path)
    data, labels = _convert_data_labels_to_torch(data, labels)
    normalize = None
    return data, labels, 'HAR', normalize

def load_mice():
    from clustpy.data import load_mice_protein
    data, labels = load_mice_protein(return_X_y=True, downloads_path=data_path)
    mu = np.mean(data)
    std = np.std(data)
    data = (data - mu)/std
    data, labels = _convert_data_labels_to_torch(data, labels)
    normalize = None
    return data, labels, 'mice', normalize

def load_coil20():
    from clustpy.data import load_coil20 as coil20
    data, labels = coil20(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "coil20", normalize

def load_coil100():
    from clustpy.data import load_coil100 as coil100
    data, labels = coil100(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "coil100", normalize

def load_weizmann():
    from clustpy.data import load_video_weizmann
    data, l = load_video_weizmann(return_X_y=True, downloads_path=data_path)
    data = data / 255
    acts = l[:, 0]
    persons = l[:, 1]
    labels = persons * len(np.unique(acts)) + acts
    normalize = 1
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, "weizmann", normalize

def load_mnist():
    from clustpy.data import load_mnist as mnist
    data, labels = mnist(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "MNIST", normalize

def load_fmnist():
    from clustpy.data import load_fmnist as fmnist
    data, labels = fmnist(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "FMNIST", normalize

def load_pendigits():
    from clustpy.data import load_pendigits as pendigits
    data, labels = pendigits(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "pendigits", normalize

def load_optdigits():
    from clustpy.data import load_optdigits as optdigits
    data, labels = optdigits(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "optdigits", normalize

def load_htru():
    from clustpy.data import load_htru2 as htru2
    data, labels = htru2(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "htru", normalize

def load_letterrecognition():
    from clustpy.data import load_letterrecognition as letterrec
    data, labels = letterrec(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "letterrecognition", normalize

def load_usps():
    from clustpy.data import load_usps as usps
    data, labels = usps(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "USPS", normalize

def load_data(loading_method):
    from sklearn.preprocessing import scale
    data, gt_labels, data_name, normalize = loading_method()
    if normalize is not None:
       data = torch.from_numpy(scale(data, axis=normalize)).float()
    return data, gt_labels, data_name
