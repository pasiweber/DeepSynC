import os
import torch
import torchvision
import numpy as np
from sklearn.datasets import make_moons, make_circles
from clustpy.data.preprocessing import ZNormalizer


data_path = "/export/share/peters57dm/Verbund/data/benchmark"
dataset_name_path_mapper = {
    'HAR' : os.path.join(data_path, 'corepts', 'har_corepts.npy'),
    'mice' : os.path.join(data_path, 'corepts', 'mice_corepts.npy'),
}


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

def load_cifar100():
    from clustpy.data import load_cifar100 as cifar100
    data, labels = cifar100(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    data = np.reshape(data,(data.shape[0],32,32,3))
    data = np.moveaxis(data,(0,1,2,3), (0,2,3,1))
    normalized_data = norm.fit_transform(data)
    normalized_data = np.reshape(normalized_data, (normalized_data.shape[0], 32*32*3))
    normalize = None
    data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return data, labels, "cifar100", normalize

def old_load_cifar100():
    from clustpy.data import load_cifar100 as cifar100
    data, labels = cifar100(return_X_y=True, downloads_path=data_path)
    data = data / 255
    normalize = 1
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, "cifar100", normalize

def load_cifar10():
    from clustpy.data import load_cifar10 as cifar10
    data, labels = cifar10(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    data = np.reshape(data,(data.shape[0],32,32,3))
    data = np.moveaxis(data,(0,1,2,3), (0,2,3,1))
    normalized_data = norm.fit_transform(data)
    normalized_data = np.reshape(normalized_data, (normalized_data.shape[0], 32*32*3))
    normalize = None
    data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return data, labels, "cifar10", normalize

def old_load_cifar10():
    from clustpy.data import load_cifar10 as cifar10
    data, labels = cifar10(return_X_y=True, downloads_path=data_path)
    data = data / 255
    normalize = 1
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, "cifar10", normalize

def load_coil20():
    from clustpy.data import load_coil20 as coil20
    data, labels = coil20(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "coil20", normalize

def old_load_coil20():
    from clustpy.data import load_coil20 as coil20
    data, labels = coil20(return_X_y=True, downloads_path=data_path)
    data = data / 255
    normalize = 1
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, 'coil20', normalize

def load_coil100():
    from clustpy.data import load_coil100 as coil100
    data, labels = coil100(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "coil100", normalize

def old_load_coil100():
    from clustpy.data import load_coil100 as coil100
    data, labels = coil100(return_X_y=True, downloads_path=data_path)
    data = data / 255
    normalize = 1
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, 'coil100', normalize

def load_cmu_faces():
    from clustpy.data import load_cmu_faces as cmu_faces
    data, labels = cmu_faces(return_X_y=True, downloads_path=data_path)
    data = data / 255
    normalize = 1
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, 'cmu_faces', normalize

def load_synth_low():
    path = f"{data_path}/low_data_100.npy"
    data, labels = np.hsplit(np.load(path), [-1])
    normalize = None
    labels = labels.reshape(-1)
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels, "synth_low", normalize

def load_synth_high():
    path = f"{data_path}/high_data_100.npy"
    data, labels = np.hsplit(np.load(path), [-1])
    normalize = None
    data, labels = _convert_data_labels_to_torch(data, labels)
    return data, labels.reshape(-1), "synth_high", normalize

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

def load_two_moons():
    my_data = make_moons(n_samples=4000, shuffle=True, noise=0.05, random_state=200)
    data = torch.from_numpy(my_data[0]).float()
    labels = torch.from_numpy(my_data[1]).float()
    normalize = None
    return data, labels, "two_moons", normalize

def load_gaussian_blobs():
    my_data1 = np.random.normal(loc=(10,-5), scale=(1,1), size=(1000,2))
    my_labels1 = np.zeros((1000))
    my_data2 = np.random.normal(loc=(20,10), scale=(2,2), size=(1000,2))
    my_labels2 = np.ones((1000))
    my_data3 = np.random.normal(loc=(0,10), scale=(3,3), size=(1000,2))
    my_labels3 = np.ones((1000))*2
    my_data = np.concatenate((my_data1, my_data2, my_data3), axis=0)
    my_labels = np.concatenate((my_labels1, my_labels2, my_labels3), axis=0)
    data = torch.from_numpy(my_data).float()
    labels = torch.from_numpy(my_labels).float()
    normalize = 0
    return data, labels, "easy_blobs", normalize

def load_two_circles():
    my_data = make_circles(n_samples=4000, shuffle=True, noise=0.05, random_state=200, factor=0.4)
    data = torch.from_numpy(my_data[0]).float()
    labels = torch.from_numpy(my_data[1]).float()
    normalize = None
    return data, labels, "two_circles", normalize

def load_runEx():
    path = os.path.join(data_path, "RunEx.csv")
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = None
    return data, labels, "runEx", normalize

def load_mnist():
    from clustpy.data import load_mnist as mnist
    data, labels = mnist(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "MNIST", normalize

def old_load_mnist():
    # setup normalization function
    mnist_mean = 0.1307
    mnist_std = 0.3081
#     normalize = torchvision.transforms.Normalize((mnist_mean,), (mnist_std,))
    # download the MNIST data set
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True)
    data = trainset.data
    # preprocess the data
    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
#     data = normalize(data)
    # Flatten from a shape of (-1, 28,28) to (-1, 28*28)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    normalize = 1
    return data, labels, "MNIST", normalize

def load_fmnist():
    from clustpy.data import load_fmnist as fmnist
    data, labels = fmnist(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "FMNIST", normalize

def old_load_fmnist():
    # setup normalization function
    fmnist_mean = 0.2860
    fmnist_std = 0.3530
#     normalize = torchvision.transforms.Normalize((fmnist_mean,), (fmnist_std,))
    # download the MNIST data set
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True)
    data = trainset.data
    # preprocess the data
    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
#     data = normalize(data)
    # Flatten from a shape of (-1, 28,28) to (-1, 28*28)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    normalize = 1
    return data, labels, "FMNIST", normalize

def load_kmnist():
    from clustpy.data import load_kmnist as kmnist
    data, labels = kmnist(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "KMNIST", normalize

def old_load_kmnist():
    # download the MNIST data set
    trainset = torchvision.datasets.KMNIST(root=data_path, train=True, download=True)
    data = trainset.data
    # preprocess the data
    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
#     data = normalize(data)
#     Flatten from a shape of (-1, 28,28) to (-1, 28*28)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    normalize = 1
    return data, labels, "KMNIST", normalize

def load_iris():
    my_data = torch.from_numpy(np.genfromtxt("data/Iris.txt", delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "iris", normalize

def load_ecoli():
    my_data = torch.from_numpy(np.genfromtxt("data/ecoli.txt", delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "ecoli", normalize

def load_wine():
    my_data = torch.from_numpy(np.genfromtxt("data/Wine.txt", delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "wine", normalize

def load_motestrain():
    my_data = torch.from_numpy(np.genfromtxt("data/MoteStrain.txt", delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "motestrain", normalize

def load_pendigits():
    from clustpy.data import load_pendigits as pendigits
    data, labels = pendigits(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "pendigits", normalize

def old_load_pendigits():
    path = os.path.join(data_path, "pendigits_adj.csv")
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=";")).float()
    data = my_data[:,1:]
    labels = my_data[:,0]
    normalize = 0
    return data, labels, "pendigits", normalize

def load_optdigits():
    from clustpy.data import load_optdigits as optdigits
    data, labels = optdigits(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "optdigits", normalize

def old_load_optdigits():
    path = os.path.join(data_path, "optdigits_adj.csv")
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=";")).float()
    data = my_data[:,1:]
    labels = my_data[:,0]
    normalize = 1
    return data, labels, "optdigits", normalize

def load_banknotes():
    my_data = torch.from_numpy(np.genfromtxt("data/banknotes.txt", delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "banknotes", normalize

def load_wafer():
    my_data = torch.from_numpy(np.genfromtxt("data/wafer.txt", delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "wafer", normalize

def load_htru():
    from clustpy.data import load_htru2 as htru2
    data, labels = htru2(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "htru", normalize

def old_load_htru():
    path = os.path.join(data_path, "htru.txt")
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "htru", normalize

def load_letterrecognition():
    from clustpy.data import load_letterrecognition as letterrec
    data, labels = letterrec(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(True)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "letterrecognition", normalize

def old_load_letterrecognition():
    path = os.path.join(data_path, 'letterrecognition.txt')
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "letterrecognition", normalize

def load_usps():
    from clustpy.data import load_usps as usps
    data, labels = usps(return_X_y=True, downloads_path=data_path)
    norm = ZNormalizer(False)
    normalized_data = norm.fit_transform(data)
    normalize = None
    normalized_data, labels = _convert_data_labels_to_torch(normalized_data, labels)
    return normalized_data, labels, "USPS", normalize

def old_load_usps():
    path = os.path.join(data_path, 'usps.csv')
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=";")).float()
    data = my_data[:,1:]
    labels = my_data[:,0]
    normalize = 1
    return data, labels, "USPS", normalize

def load_mini_mnist():
    data, labels, _, normalize = load_mnist()
    selection = labels < 3
    data = data[selection]
    labels = labels[selection]
    return data, labels, "Mini-MNIST", normalize

def load_example():
    path = os.path.join(data_path, "tmp6.csv")
    my_data = torch.from_numpy(np.genfromtxt(path, delimiter=",")).float()
    data = my_data[:,:-1]
    labels = my_data[:,-1]
    normalize = 0
    return data, labels, "example", normalize

def load_data(loading_method):
    from sklearn.preprocessing import scale
    data, gt_labels, data_name, normalize = loading_method()
    if normalize is not None:
       data = torch.from_numpy(scale(data, axis=normalize)).float()
    return data, gt_labels, data_name
