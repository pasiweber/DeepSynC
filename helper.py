#region imports
import os
import pickle
import torch
import torchvision
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from scipy.spatial.distance import cdist
# core points clustering methods
from SHiP import SHiP
# clustpy competitors
from clustpy.data.preprocessing import ZNormalizer
from clustpy.deep.enrc import ACeDeC
from clustpy.deep.dcn import DCN
from clustpy.deep.ddc_n2d import DDC
from clustpy.deep.dec import DEC
from clustpy.deep.dipdeck import DipDECK
from clustpy.deep.dkm import DKM
from clustpy.deep.dec import IDEC
# sklearn competitors
from sklearn.cluster import ( 
    HDBSCAN,
    AffinityPropagation,
    MeanShift
)
#endregion


data_path = "/export/share/peters57dm/Verbund/data/benchmark"
dataset_name_path_mapper = {
    'HAR' : os.path.join(data_path, 'corepts', 'har_corepts.npy'),
    'mice' : os.path.join(data_path, 'corepts', 'mice_corepts.npy'),

}


#region DS
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
    if not normalize is None:
       data = torch.from_numpy(scale(data, axis=normalize)).float()
    return data, gt_labels, data_name
#endregion


#region Paper Plot
# plots
def plot_2d_dataset(data, labels, core_points_mask=None, centers=None, fixed_scales = False, save=None):
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        if centers is not None:
            centers = pca.transform(centers)
    _, ax = plt.subplots(figsize=(10,10))
    ax.scatter(data[:, 0], data[:, 1], c=labels, label='Data points', alpha=0.6)
    if not core_points_mask is None:
        ax.scatter(data[np.where(np.diag(core_points_mask)==1),0],data[np.where(np.diag(core_points_mask)==1),1],c = 'r')
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], s = 10, c="red")
        for j in range(len(centers)):
            plt.text(centers[j,0], centers[j,1], str(j))
    if fixed_scales is True:
        ax.set(xlim=(-10, 10), ylim=(-10, 10))
    else:
        ax.axis("equal")
    if not save is None:
        plt.savefig(save)
    plt.show()
    plt.close()
def plot_and_save(values, title, xlabel, ylabel, save_path, legend_label=None, grid=True):
    """
    Plots a list of values and takes title, axes names, and legend information as input.
    Saves the plot as an image to the specified path.
    
    Parameters:
    - values: List of values to plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - legend_label: Label for the legend.
    - save_path: Path to save the image (e.g., 'plot.png').
    - grid: Boolean to show the grid or not
    """
    if legend_label is None:
        plt.plot(values)
    else :
        plt.plot(values, label=legend_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not legend_label is None:
        plt.legend()
    plt.grid(grid)
    
    # Save the plot to the given path
    plt.savefig(save_path)
    
    # Optionally, you can show the plot as well
    plt.show()
def plot_and_save_two_lists(list1, list2, title, xlabel, ylabel, legend_label1, legend_label2, save_path):
    """
    Plots two lists of values on the same figure and takes title, axes names, and legend information as input.
    Saves the plot as an image to the specified path.
    
    Parameters:
    - list1: First list of values to plot.
    - list2: Second list of values to plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - legend_label1: Label for the first list in the legend.
    - legend_label2: Label for the second list in the legend.
    - save_path: Path to save the image (e.g., 'plot.png').
    """
    # Plot the first list
    plt.plot(list1, label=legend_label1, color='b')  # Blue color for the first list
    # Plot the second list
    plt.plot(list2, label=legend_label2, color='r')  # Red color for the second list
    
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add legend
    plt.legend()
    
    # Add grid for better visualization
    plt.grid(True)
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Optionally, you can show the plot as well
    plt.show()
def plot_3d_dataset(data, labels, centers=None):
    if data.shape[1] < 3:
        return
    print("3D Plot")
    if data.shape[1] > 3:
        pca = PCA(n_components=3)
        data = pca.fit_transform(data)
        centers = pca.transform(centers)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], zs=data[:, 2], zdir='z', s=10, c=labels)
    if centers is not None:
        for i in range(len(centers)):
            ax.text(centers[i,0], centers[i,1], centers[i, 2], str(i))
    plt.show()
def denormalize(tensor:torch.Tensor, mean:float=0.1307, std:float=0.3081)->torch.Tensor:
    """
    This applies an inverse z-transformation and reshaping to visualize
    the mnist images properly.
    """
    pt_std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    pt_mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    return (tensor.mul(pt_std).add(pt_mean).view(-1, 1, 28,28) * 255).int().detach()
def plot_images(images:torch.Tensor, pad:int=0):
    """Aligns multiple images on an N by 10 grid"""
    def imshow(img):
        plt.figure(figsize=(10, 20))
        npimg = img.numpy()
        npimg = np.array(npimg)
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), vmin=0, vmax=1)

    imshow(torchvision.utils.make_grid(images, pad_value=255, normalize=False, padding=pad, nrow=10))
    plt.show();
def print_accuracies(gt_labels, epoch_labels, return_results=True):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari
    labeled_pts_mask = epoch_labels > -1
    lbld_pts_score = ami(gt_labels[labeled_pts_mask],
                                    epoch_labels[labeled_pts_mask])
    all_pts_score = ami(gt_labels, epoch_labels)
    lbld_pts_ari = ari(gt_labels[labeled_pts_mask],
                        epoch_labels[labeled_pts_mask])
    all_pts_ari = ari (gt_labels, epoch_labels)
    print("Labeled pts AMI = ", lbld_pts_score)
    print("All pts AMI = ", all_pts_score)
    print("labeled pts ARI = ", lbld_pts_ari)
    print("All pts ARI = ", all_pts_ari)
    
    if return_results:
        return lbld_pts_score, all_pts_score, lbld_pts_ari, all_pts_ari
def update_eval_tracker(gt_labels, epoch_labels, eval_tracker):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari
    labeled_pts_mask = epoch_labels > -1
    lbld_pts_ami = ami(gt_labels[labeled_pts_mask],
                                    epoch_labels[labeled_pts_mask])
    all_pts_ami = ami(gt_labels, epoch_labels)
    lbld_pts_ari = ari(gt_labels[labeled_pts_mask],
                        epoch_labels[labeled_pts_mask])
    all_pts_ari = ari (gt_labels, epoch_labels)

    eval_tracker.ami_labeled.append(lbld_pts_ami)
    eval_tracker.ami_total.append(all_pts_ami)
    eval_tracker.ari_labeled.append(lbld_pts_ari)
    eval_tracker.ari_total.append(all_pts_ari)
    eval_tracker.no_labeled_pts.append(int(np.sum(labeled_pts_mask)))
    eval_tracker.set_predicted_labels(epoch_labels.tolist())
    return eval_tracker
def save_dict_as_json(json_object, file_path):
    import json
    with open(file_path, 'w') as json_file:
        json.dump(json_object, json_file, indent=4)
def load_json_as_dict(file_path):
    import json
    with open(file_path, "r") as json_file:
        return json.load(json_file)
def create_gif_from_directory(directory_path, output_filename='output.gif', duration=500):
    import re
    from PIL import Image
    """
    Create a GIF from all images in the specified directory, sorted by numeric order in filenames.
    Only includes images that follow the pattern of three digits (e.g., 000.jpeg, 001.jpeg, ...).

    Args:
    - directory_path (str): Path to the directory containing the images.
    - output_filename (str): Name of the output GIF file. Default is 'output.gif'.
    - duration (int): Duration for each frame in milliseconds. Default is 500ms.

    Returns:
    - None
    """
    # Regex pattern to match image filenames of the form "nnn.jpeg" or "nnn.jpg"
    pattern = r'^\d{3}\.(jpeg|jpg|png)$'

    # Get a list of image files in the directory that match the pattern
    image_files = [f for f in os.listdir(directory_path) if re.match(pattern, f)]

    # Sort the files by the numeric part of the filename (e.g., 000, 001, 002...)
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Create a list to store the images
    images = []

    # Open each image and append to the images list
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Check if there are images to process
    if not images:
        print("No valid images found in the specified directory.")
        return

    # Create the GIF from the list of images
    images[0].save(output_filename, save_all=True, append_images=images[1:], loop=0, duration=duration)
#endregion


#region kmeans
def kmeans_clustering(data, n_clusters, base_path, data_name=None, normalize = None, input_centers=None, random_state=7):
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
def plot_2d_kmeans_result(data, gt_labels, cluster_labels, centers, pairplot=False):
    all_colors = ['darkgray', 'red', 'peru',
                'yellow', 'olive', 'lawngreen', 'lightsalmon', 'cyan', 'darkorange', 'blue',
                'darkorchid', 'springgreen', 'fuchsia', 'firebrick', 'crimson', 'pink', 'teal', 'gold']
    if pairplot:
        import seaborn as sns
        import pandas as pd

        df = pd.DataFrame(data)
        nr_clusters = len(np.unique(cluster_labels))
        columns = df.columns.values
        df["labels"] = cluster_labels
        sns.set(style="ticks")
        colors = all_colors * (nr_clusters // len(all_colors)) + all_colors[:(nr_clusters % len(all_colors))]
        sns.pairplot(df, hue="labels", vars=columns, diag_kind="hist", palette=colors)
    else:
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)
            centers = pca.transform(centers)

        markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'X', 'D']
        fig, ax = plt.subplots(figsize=(10,10))
        unique_labels = np.unique(gt_labels)
        for i, l in enumerate(unique_labels):
            ax.scatter(data[gt_labels == l,0], data[gt_labels == l,1], c=[all_colors[i % len(all_colors)] for i in cluster_labels[gt_labels == l]], s=6, marker = markers[i % len(markers)])
        ax.scatter(centers[:,0], centers[:,1], c="red")
        for i in range(len(centers)):
            ax.text(centers[i,0], centers[i,1], str(i))
        ax.axis("equal")
        #ax.set(xlim=(-2,2), ylim=(-2,2))
    plt.show()
# https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/
#endregion


#region extras
def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)
def get_clust_confusion_matrix(gt, pred):
#     cm = confusion_matrix(gt, pred)
#     indexes = linear_assignment(_make_cost_m(cm))
#     js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
#     cm2 = cm[:, js]
#     return cm2
    gt_clusters = np.unique(gt)
    pred_clusters = np.unique(pred)
    conf_matrix = np.zeros((len(gt_clusters), len(pred_clusters)), dtype=int)
    for i, gt_label in enumerate(gt_clusters):
        point_labels = pred[gt == gt_label]
        labels, cluster_sizes = np.unique(point_labels, return_counts=True)
        for j, pred_label in enumerate(labels):
            conf_matrix[i, np.argwhere(pred_clusters == pred_label)[0][0]] = cluster_sizes[j]
    # Create Plot
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(conf_matrix, cmap="YlGn")
    for i in range(len(gt_clusters)):
        for j in range(len(pred_clusters)):
            ax.text(j, i, conf_matrix[i, j],
                           ha="center", va="center", color="black")
    plt.show()
    return conf_matrix
#endregion


#region DL methods
def int_to_one_hot(label_tensor, n_labels):
    onehot = torch.zeros([label_tensor.shape[0], n_labels], dtype=torch.float, device=label_tensor.device)
    onehot.scatter_(1, label_tensor.unsqueeze(1).long(), 1.0)
    return onehot
def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    embeddings = []
    labels = []
    for batch, batch_labels, ids in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0)
def decode_batchwise(dataloader, model, device):
    """ Utility function for decoding the whole data set in a mini-batch fashion
    """
    decodings = []
    for batch, labels, ids in dataloader:
        batch_data = batch.to(device)
        decodings.append(model(batch_data).detach().cpu())
    return torch.cat(decodings, dim=0).numpy()
def get_train_and_testloader(data, labels, batch_size):
    # create a Dataloader to train the autoencoder in mini-batch fashion
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data, labels, torch.arange(0, len(labels)))),
                                                  batch_size=batch_size,
                                                  # sample random mini-batches from the data
                                                  shuffle=True,
                                                  drop_last=False)
    # create a Dataloader to test the autoencoder in mini-batch fashion (Important for validation)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data, labels, torch.arange(0, len(labels)))),
                                             batch_size=batch_size,
                                             # Note that we deactivate the shuffling
                                             shuffle=False,
                                             drop_last=False)
    return trainloader, testloader
def get_eps_neighbourhood(data, eps):
    n = data.shape[0]
    squared_diffs = (data.unsqueeze(0) - data.unsqueeze(1)).pow(2).sum(2)
    #NB_eps_mask = torch.Tensor(np.double(squared_diffs < eps)-np.eye(n)) # exclude point itself
    NB_eps_mask = torch.Tensor(np.double(squared_diffs < eps)) # include point becuase empty neighbourhood produces nan
    return squared_diffs, NB_eps_mask
#Kuramoto Order Parameter with eps neighbourhood
def calc_KOP_eps(data,eps):
    n = data.shape[0]
    di,mask = get_eps_neighbourhood(data,eps)
    r_c = 1/n*(1/(mask.sum(0))*(np.exp(-di)*mask).sum(0)).sum()
    return r_c
# distances to k nearest neighbours and their indices
def get_k_nns(data, k):
    squared_diffs = (data.unsqueeze(0) - data.unsqueeze(1)).pow(2).sum(2)
    knn_distance_based = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(squared_diffs)
    distances, indices = knn_distance_based.kneighbors(squared_diffs)
    return distances, indices
#Kuramoto Order Parameter with k NNs
def calc_KOP_nn(data,k):
    n = data.shape[0]
    dists , k_nn_ind = get_k_nns(data,k)
    r_c = np.exp(-dists).sum()
    r_c *= 1/(n*k)
    return r_c
def pretrain_model(device, pretrained_model_path, model, data, trainloader, optimizer, loss_fn, training_iterations = 30, verbose=True):
    # load model to device
    model.to(device)

    # start training
    i = 0
    # training loop
    while(i < training_iterations):
        for batch,_, _ in trainloader:
            # load batch on device
            batch_data = batch.to(device)

            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch_data)

            # reset gradients from last iteration
            optimizer.zero_grad()
            # calculate gradients and reset the computation graph
            loss.backward()
            # update the internal params (weights, etc.)
            optimizer.step()
#             if i > training_iterations:
#                 print("Stop training")
#                 break
        if verbose:
            print(f"Iteration {i+1}/{training_iterations} - Reconstruction loss: {loss.item():.6f}")
        i += 1
    # save model
    torch.save(model.state_dict(), pretrained_model_path)
    return model
def load_pretrained_model(model, pretrained_model_path, device, work_on_copy = False):
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    if hasattr(model, "fitted"):
        model.fitted=True
    model.work_on_copy = work_on_copy
    return model
def calculate_squared_differences_vectorized(data):
    """
    Function to calculate the squared differences between all pairs of data points in a vectorized manner.

    Parameters:
    - data: A numpy array of shape (n_samples, n_features), representing the dataset.

    Returns:
    - squared_diff_matrix: A (n_samples, n_samples) numpy array containing the squared differences between all pairs.
    """
    squared_diffs = (data.unsqueeze(0) - data.unsqueeze(1)).pow(2).sum(2)

    return squared_diffs.detach().cpu().numpy()
def find_core_points(squared_diffs, k):
    """
    Function to find core points in regions of high density.

    Parameters:
    - data: A numpy array of shape (n_samples, n_features), representing the dataset.
    - eps: The distance within which to consider neighbors.
    - min_samples: The minimum number of points required to be considered a core point.

    Returns:
    - core_points: A numpy array containing the actual coordinates of the core points.
    """
    # Fit NearestNeighbors to find neighbors within eps distance
    n = len(squared_diffs)
    knn_distance_based = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(squared_diffs)
    distances, indices = knn_distance_based.kneighbors(squared_diffs)
    density_threshold = np.median(distances[:,k-1])
    mask = np.ones((n, k))
    mask[distances[:,k-1]>density_threshold,:] = 0

    core_mask_cpu = np.zeros((n,n))
    for j in range(0, n):
        for l in range(0, n):
            if (mask[j,0] == 1 and mask[l,0] == 1):
                core_mask_cpu[j,l] = 1
    # final_mask_cpu = core_mask_cpu*(squared_diffs<(density_threshold))
    return core_mask_cpu
# precomputes the k-th nearest neighbor distance (core_dist / kappa) and reuses this within every T% nearest neighbor subset
def find_local_core_points_same(data, k, percent):
    from sklearn.metrics import pairwise_distances

    n = data.shape[0]
    subset = int(np.floor(n * percent))

    p_dist = pairwise_distances(data, metric="euclidean")
    core_dists = np.partition(p_dist, k - 1, axis=0)[k - 1]

    nn = np.argpartition(p_dist, subset, axis=1)[:, :subset]
    # nn_mask = np.tile(np.arange(n).reshape(-1, 1), (1, subset))
    # nn = nn[nn != nn_mask].reshape(n, subset - 1)
    refined_medians = np.median(core_dists[nn], axis=1)
    med_th = np.median(refined_medians)

    vector_mask = core_dists < refined_medians
    core_points_mask = np.outer(vector_mask, vector_mask).astype(int)
    return core_points_mask, med_th
def find_local_core_points_fast(data, k, percent):
    n = data.shape[0]
    subset = int(np.floor(n * percent))

    knn = NearestNeighbors(n_neighbors=subset, metric="euclidean").fit(data)
    distances, indices = knn.kneighbors(data)
    
    # Compute k-nearest neighbors within each subset
    refined_medians = np.zeros(n)
    for j in range(n):
        j_neighbors = data[indices[j], :]
        knn_j = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(j_neighbors)
        distances_j, _ = knn_j.kneighbors(j_neighbors)
        refined_medians[j] = np.median(distances_j[:, k-1])
    
    furthest_neighbor_points_distances = distances[:, k-1]
    vector_mask = furthest_neighbor_points_distances < refined_medians
    
    core_points_mask = np.outer(vector_mask, vector_mask).astype(int)
    core_threshold = np.median(refined_medians)
    
    return core_points_mask, core_threshold
def find_local_core_points(data, k, percent):
    # Fit NearestNeighbors to find neighbors within eps distance
    n = data.shape[0]
    subset = int(np.floor(n*percent))
    knn_distance_based = NearestNeighbors(n_neighbors=subset, metric="euclidean").fit(data)
    distances, indices = knn_distance_based.kneighbors(data)
    mask = np.zeros(n)
    medians = np.zeros(n)
    for j in range(0,n):
        j_neighbours = data[indices[j,:],:]
        knn_distance_based_j = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(j_neighbours)
        distances_j, _ = knn_distance_based_j.kneighbors(j_neighbours)
        density_threshold = np.median(distances_j[:,k-1])
        medians[j] = density_threshold
        if distances[j,k-1] < density_threshold: # or distances_j[0,k-1]
            mask[j] = 1 
    #x = np.where(mask==1)
    #core_point_ind= x[0]
    core_mask_cpu = np.zeros((n,n))
    for r in range(0, n):
        for l in range(0, n):
            if (mask[r] == 1 and mask[l] == 1):
                core_mask_cpu[r,l] = 1
    core_thresh = np.median(medians)
    return core_mask_cpu, core_thresh
#endregion


#region Loss Functions
def attraction_repelling_loss(model, batch_data, iteration_labels, losses_tracker, device):
    embedded = model.encode(batch_data)
    unique_labels = np.unique(iteration_labels)
    unique_labels = unique_labels[unique_labels > -1]
    if len(unique_labels) == 0:
        print("No labels in this batch")
        dont_propagate = True
        loss = 0
        return loss, dont_propagate, losses_tracker

    dont_propagate = False
    attract_loss = 0
    repel_loss = 0
    for l in unique_labels:
        # calculate attraction loss
        label_mask = iteration_labels == l
        label_pts = embedded[label_mask]
        label_square_diffs = (label_pts.unsqueeze(0) - label_pts.unsqueeze(1)).pow(2).sum(2)
        label_weights = 1 - (label_square_diffs / torch.max(label_square_diffs))
        n_label_pts = len(label_pts)
        _att_val = 1/n_label_pts**2 * torch.exp(-label_square_diffs*label_weights).sum()
        if not torch.isnan(_att_val) :
            attract_loss += 1 - _att_val

        # calculate repeling loss
        other_pts_mask = np.logical_and(iteration_labels != l, iteration_labels > -1)
        other_pts = embedded[other_pts_mask]
        other_pts_square_diffs = (label_pts.unsqueeze(0) - other_pts.unsqueeze(1)).pow(2).sum(2)
        label_weights = 1 # fixed label weight
        no_other_pts = len(other_pts_square_diffs)
        if no_other_pts == 0:
            _rep_val = torch.tensor(0)
        else :
            _rep_val = 1/(no_other_pts)**2 * torch.exp(-other_pts_square_diffs * label_weights).sum()
        
        if not torch.isnan(_rep_val) :
            repel_loss += _rep_val

    loss = 1/len(unique_labels) * (attract_loss + repel_loss)

    if not isinstance(loss, int): 
        losses_tracker.total.append(loss.detach().cpu().numpy().item())
    else :
        losses_tracker.total.append(loss)

    if not isinstance(repel_loss, int):
        losses_tracker.repel.append(repel_loss.detach().cpu().numpy().item())
    else :
        losses_tracker.repel.append(repel_loss)

    if not isinstance(attract_loss, int):
        losses_tracker.attract.append(attract_loss.detach().cpu().numpy().item())
    else :
        losses_tracker.attract.append(attract_loss)

    return loss, dont_propagate, losses_tracker
class AttractionRepellingLossTracker:
    def __init__(self, total=None, repel=None, attract=None):
        # Initialize attributes with default empty lists if None is provided
        self.total = total if total is not None else []
        self.repel = repel if repel is not None else []
        self.attract = attract if attract is not None else []

    def to_dict(self):
        # Converts the instance attributes into a dictionary
        return {
            'total': self.total,
            'repel': self.repel,
            'attract': self.attract
        }
    
    def __repr__(self):
        return f"AttractionRepellingLossTracker(total={self.total}, repel={self.repel}, attract={self.attract})"
class AESyncLossTracker:
    def __init__(self, total=None, ae=None, sync=None):
        # Initialize attributes with default empty lists if None is provided
        self.total = total if total is not None else []
        self.ae_loss = ae if ae is not None else []
        self.sync_loss = sync if sync is not None else []

    def to_dict(self):
        # Converts the instance attributes into a dictionary
        return {
            'total': self.total,
            'ae': self.ae_loss,
            'sync': self.sync_loss
        }

    def __repr__(self):
        return f"AESyncLossTracker(Total_Loss={self.total}, AEReconstruction={self.ae}, Sync_Loss={self.sync})"
def _sync_loss(model, batch_data, current_batch_labels, device):
    def get_scaled_outlier_dists(out_dists):
        #input: distance matrix of a batch masked for the distances of outliers 
        #output: scaled distances for outliers based on distance 
        max_vec = np.max(out_dists,1)
        s = max_vec/4
        s = np.transpose(np.repeat([s], out_dists.shape[0], axis=0))
        #print(s)
        x = np.multiply(out_dists,1/s)
        return np.exp(-1/2*np.power(x,2))
    embedded = model.encode(batch_data)
    squared_diffs = (embedded.unsqueeze(0) - embedded.unsqueeze(1)).pow(2).sum(2)
    squared_diffs_cpu = squared_diffs.detach().cpu().numpy()
    n = len(current_batch_labels)
    outliers = np.eye(n)
    for j in range(0,n):
        if current_batch_labels[j] < 0:
            outliers[:,j] = 1 
            outliers[j,:] = 1
    outlier_dists = squared_diffs_cpu*outliers
    scaled_outlier_weights = get_scaled_outlier_dists(outlier_dists)
    scaled_weights = scaled_outlier_weights
    for k in range(0,n):
        if current_batch_labels[k] >= 0:
            for l in range(0,n):
                if current_batch_labels[l] >= 0:
                    if current_batch_labels[k] == current_batch_labels[l]:
                        scaled_weights[k,l] = 1 
                        scaled_weights[l,k] = 1
                    else:
                        scaled_weights[k,l] = 0 
                        scaled_weights[l,k] = 0
    scaled_weights_torch = torch.from_numpy(scaled_weights).to(device)
    sync =  (1/n**2)*(torch.exp(-squared_diffs*scaled_weights_torch)).sum(0).sum()
    sync_loss = 1 - sync
    return sync_loss
def _reconstrunction_mse_loss(model, batch_data):
    embedded = model.encode(batch_data)
    reconstruction = model.decode(embedded)
    loss_fun = torch.nn.MSELoss()
    ae_loss = loss_fun(reconstruction, batch_data)
    return ae_loss
def ae_sync_loss(model, batch_data, current_batch_labels, losses_tracker, device):
    ae_loss = _reconstrunction_mse_loss(model, batch_data)
    sync_loss = _sync_loss(model, batch_data, current_batch_labels, device)
    loss = ae_loss + sync_loss
    losses_tracker.ae_loss.append(ae_loss.detach().cpu().numpy().item())
    losses_tracker.sync_loss.append(sync_loss.detach().cpu().numpy().item())
    losses_tracker.total.append(loss.detach().cpu().numpy().item())
    dont_propagate = False
    return loss, dont_propagate, losses_tracker
class EvaluationTracker:
    def __init__(self, ari_labeled=None, ari_total=None,
                 ami_labeled=None, ami_total=None,
                 no_labeled_pts=None, dataset_size=None):
        # Initialize attributes with default empty lists if None is provided
        self.ari_labeled = ari_labeled if ari_labeled is not None else []
        self.ari_total = ari_total if ari_total is not None else []
        self.ami_labeled = ami_labeled if ami_labeled is not None else []
        self.ami_total = ami_total if ami_total is not None else []
        self.no_labeled_pts = no_labeled_pts if no_labeled_pts is not None else []
        self.dataset_size = dataset_size
        self.predicted_labels = None

    def to_dict(self):
        # Converts the instance attributes into a dictionary
        return {
            'ari_labeled': self.ari_labeled,
            'ari_total': self.ari_total,
            'ami_labeled': self.ami_labeled,
            'ami_total': self.ami_total,
            'predicted_labels' : self.predicted_labels,
            'no_labeled_pts': self.no_labeled_pts,
            'dataset_size': int(self.dataset_size)
        }
    
    def set_predicted_labels(self, predictions):
        self.predicted_labels = predictions

    def __repr__(self):
        return (f"EvaluationTracker(ari_labeled={self.ari_labeled}, "
                f"ari_total={self.ari_total}, ami_labeled={self.ami_labeled}, "
                f"ami_total={self.ami_total})", f"labeled/total={self.no_labeled_pts}/{self.dataset_size}")
#endregion


#region assign labels
def get_high_cof_labels(prediction_matrix, n, crop):
    cols_to_check = prediction_matrix
    if not crop is None:
        cols_to_check = prediction_matrix[:, -n:]
    else : # iteration number < n_check
        # initialize with all zeros
        high_confidence_labels = np.zeros(prediction_matrix.shape[0], dtype=bool)
        # all labeled points are high confidence labels
        high_confidence_labels[cols_to_check[:, 0] >= 0] = True 
        return high_confidence_labels
    i = 0
    all_equal = cols_to_check[:, i] == cols_to_check[:, i+1]
    while i+1 < cols_to_check.shape[1]:
        check = cols_to_check[:, i] == cols_to_check[:, i+1]
        all_equal = np.logical_and(all_equal, check)
        i += 1
    custom_labels_mask = prediction_matrix[:, -2] < 0
    all_equal[custom_labels_mask] = False
    return all_equal
def check_equality_in_n_consequtive_cols(prediction_matrix, n, crop):
    cols_to_check = prediction_matrix
    if not crop is None:
        cols_to_check = prediction_matrix[:, crop-n:crop]
    i = 0
    all_equal = cols_to_check[:, i] == cols_to_check[:, i+1]
    while i+1 < cols_to_check.shape[1]:
        check = cols_to_check[:, i] == cols_to_check[:, i+1]
        all_equal = np.logical_and(all_equal, check)
        i += 1
    custom_labels_mask = prediction_matrix[:, i] < 0
    all_equal[custom_labels_mask] = False
    return all_equal
def most_frequent(arr):
    n = len(arr)
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]

    return res
def find_cluster_intersections(labels1, labels2):
    # Check if the two lists are of the same length
    if len(labels1) != len(labels2):
        raise ValueError("Both label lists must be of the same length")

    # Dictionary to store elements of each cluster for both label lists
    clusters1 = defaultdict(list)
    clusters2 = defaultdict(list)

    # Group elements by their cluster labels in labels1
    for idx, label in enumerate(labels1):
        clusters1[label].append(idx)

    # Group elements by their cluster labels in labels2
    for idx, label in enumerate(labels2):
        clusters2[label].append(idx)

    # Find intersections between clusters
    intersections = []
    for cluster1, elements1 in clusters1.items():
        for cluster2, elements2 in clusters2.items():
            # Find common elements between two clusters
            common_elements = set(elements1).intersection(elements2)
            if common_elements:
                intersections.append((cluster1, cluster2, list(common_elements)))

    return intersections
def assign_unified_labels(labels1, labels2):
    intersections = find_cluster_intersections(labels1, labels2)

    # Create a unified label list initialized with -1 (or any placeholder for unassigned)
    unified_labels = [-1] * len(labels1)

    # Assign a new unified label to each intersection
    current_label = 0
    for cluster1, cluster2, common_elements in intersections:
      if cluster1 == -1 or cluster2 == -1:
                continue
      for idx in common_elements:
          unified_labels[idx] = current_label
      current_label += 1
    unified_labels = np.array(unified_labels)
    for label in set(unified_labels):
        if label == -1:
            continue
        mask = unified_labels == label
        mask.astype(int)
        if sum(mask) == 1:
            # assign -1 (outlier) if labels disagree
            unified_labels[mask] = -1
    final_label_mapping = {-1: -1}
    label_max = -1
    for label in set(unified_labels):
        if not label in final_label_mapping:
            label_max += 1
            final_label_mapping[label] = label_max

    # # allow change in labels if previously was defined as outlier
    # unified_labels[labels1 == -1] = labels2[labels1 == -1]
    # # if a point is considered an outlier while it was not before, ignore it.
    # unified_labels[labels2 == -1] = labels1[labels2 == -1]

    # final labeling assignment
    for k, v in final_label_mapping.items():
      if k == -1:
        unified_outlier_mask = unified_labels == k
        labels2_outlier_mask = labels2 == k
        labels1_outlier_mask = labels1 == k

        # In any case
        # it must be an outlier in the unified labels final results

        # case_0: if outlier in labels1 and labels2 => set as outlier
        case_0_mask = np.logical_and(labels1_outlier_mask,
                                     labels2_outlier_mask)
          # Ensure that those indicies are already classified as outliers
          # in the unified labels
        case_0_mask = np.logical_and(case_0_mask, unified_outlier_mask)
        unified_labels[case_0_mask] = -1

        # case_1: if outlier in labels2 only => set label of labels1
        case_1_mask = np.logical_and(labels2_outlier_mask == True,
                                     labels1_outlier_mask == False)
          # Ensure that those indicies are already classified as outliers
          # in the unified labels
        case_1_mask = np.logical_and(case_1_mask, unified_outlier_mask)

          # assign the most frequent label
          # This trick is done because we can't assume that
          # the label number is the same for each cluster
          # across different iterations
        case_1_labels1_possible_values = labels1[case_1_mask]
        case_1_labels1_most_frequent = most_frequent(case_1_labels1_possible_values)
        case_1_labels1_most_frequent_mask = labels1 == case_1_labels1_most_frequent

        possible_values = unified_labels[case_1_labels1_most_frequent_mask]
        value = most_frequent(possible_values)
        unified_labels[case_1_mask] = value


        # case_2: if outlier in labels1 only => set label of labels2
        case_2_mask = np.logical_and(labels1_outlier_mask == True,
                                          labels2_outlier_mask == False)

          # Ensure that those indicies are already classified as outliers
          # in the unified labels
        case_2_mask = np.logical_and(case_2_mask, unified_outlier_mask)

          # assign the most frequent label
        case_2_labels2_possible_values = labels2[case_2_mask]
        case_2_labels2_most_frequent = most_frequent(case_2_labels2_possible_values)
        case_2_labels2_most_frequent_mask = labels2 == case_2_labels2_most_frequent

        possible_values = unified_labels[case_2_labels2_most_frequent_mask]
        value = most_frequent(possible_values)
        unified_labels[case_2_mask] = value

        # case_3: if not outlier in any of them set label of labels2
        case_3_mask = np.logical_and(labels1_outlier_mask == False,
                                          labels2_outlier_mask == False)

          # Ensure that those indicies are already classified as outliers
          # in the unified labels
        case_3_mask = np.logical_and(case_3_mask, unified_outlier_mask)

        # assign the most frequent label
        case_3_labels2_possible_values = labels2[case_3_mask]
        case_3_labels2_most_frequent = most_frequent(case_3_labels2_possible_values)
        case_3_labels2_most_frequent_mask = labels2 == case_3_labels2_most_frequent

        possible_values = unified_labels[case_3_labels2_most_frequent_mask]
        value = most_frequent(possible_values)
        unified_labels[case_3_mask] = value

      # assign the unified labels
      mask = unified_labels == k
      unified_labels[mask] = v

    return unified_labels
def get_high_conf_labels(prediction_matrix, n_check, iter_n):
  cols_to_check = prediction_matrix.copy()
  crop = iter_n < n_check
  if crop:
    cols_to_check = cols_to_check[:,:iter_n]
  else:
    cols_to_check = cols_to_check[:, iter_n-n_check:iter_n]
  i = 0
  # handle boundary conditions
  if i+1 < cols_to_check.shape[1]:
    all_equal = cols_to_check[:, i] == cols_to_check[:, i+1]
  else:
    all_equal = cols_to_check[:, i] > -1
  # if labels disagree in second epoch, it is not reliable enough
  while i+1 < cols_to_check.shape[1]:
    check = cols_to_check[:, i] == cols_to_check[:, i+1]
    all_equal = np.logical_and(all_equal, check)
    i += 1
  custom_labels_mask = cols_to_check[:, -1] < 0
  all_equal[custom_labels_mask] = False
  return all_equal
def distance_nearest_neighbor(labeled_points, labels, unlabeled_points, distance_threshold):
    # in future implementation, we need to include the option of mahalanobis distance from
    # the centroid of each cluster in the labeled points.
    """
    Assign labels to unlabeled points based on the nearest labeled points
    while keeping far points unlabeled.

    Parameters:
    - labeled_points: np.ndarray of shape (n_labeled, n_features)
    - labels: np.ndarray of shape (n_labeled,)
    - unlabeled_points: np.ndarray of shape (n_unlabeled, n_features)
    - distance_threshold: float, the threshold distance for assigning labels
    Returns:
    - assigned_labels: np.ndarray of shape (n_unlabeled,), assigned labels for unlabeled points
    """
    # Compute distances from unlabeled points to labeled points
    distances = cdist(unlabeled_points, labeled_points)
    ###
      # we can use the mahalanobis distance here to consider the data distribution in the calc
    ###

    # Initialize an array to store assigned labels
    assigned_labels = np.full(unlabeled_points.shape[0], -1)  # -1 for unlabeled
    # Assign labels to the nearest labeled points if within the threshold
    for i in range(distances.shape[0]):
        nearest_index = np.argmin(distances[i])  # Index of the nearest labeled point
        nearest_distance = distances[i, nearest_index]

        if nearest_distance < distance_threshold:
            assigned_labels[i] = labels[nearest_index]
    return assigned_labels
def calc_distance_threshold(data_complete, all_labels):
  mask = all_labels > -1
  labels = all_labels[mask]
  data = data_complete[mask]
  labels_set = set(labels)
  distance_medians = []
  for label in labels_set:
    cluster_mask = labels == label
    cluster_pts = data[cluster_mask]
    squared_diffs = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2)
    cluster_median_distance = np.median(squared_diffs)
    distance_medians.append(cluster_median_distance)
  return np.min(distance_medians) * 0.05
def mahalanobis_assign_unlabeled_points(data, labels):
    from scipy.spatial.distance import mahalanobis
    from scipy.spatial.distance import euclidean
    unique_labels = set(labels) - {-1}  # Get unique cluster labels excluding -1

    # Compute cluster centroids, covariance matrices, and thresholds
    centroids = {}
    cov_matrices = {}
    inv_cov_matrices = {}
    thresholds = {}

    outlier_points = data[labels == -1]
    if len(outlier_points) == 0: # all points are labeled
        return labels # return the same labels

    for label in unique_labels:
        cluster_points = data[labels == label]
        centroids[label] = np.mean(cluster_points, axis=0)
        cov_matrices[label] = np.cov(cluster_points, rowvar=False)

        if cov_matrices[label].ndim < 2: # to handle zero dimensional arrays when no points for a certain label.
            thresholds[label] = 0
            continue
        # Compute inverse covariance matrix (handle singular matrices)
        try:
            inv_cov_matrices[label] = np.linalg.inv(cov_matrices[label])
        except np.linalg.LinAlgError:
            inv_cov_matrices[label] = np.linalg.pinv(cov_matrices[label])  # Use pseudo-inverse if singular

        # Compute the Mahalanobis distance for each point in the cluster to its centroid
        distances = [mahalanobis(p, centroids[label], inv_cov_matrices[label]) for p in outlier_points]
        thresholds[label] = np.percentile(distances, 10)
        # distances = [mahalanobis(p, centroids[label], inv_cov_matrices[label]) for p in cluster_points]
        # thresholds[label] = np.max(distances) * 1.05

    # Assign unlabeled points (-1) to the nearest centroid based on Mahalanobis distance
    for i, point in enumerate(data):
        if labels[i] == -1:
            min_distance = float('inf')
            assigned_label = -1

            for label in unique_labels:
                try :
                    distance = mahalanobis(point, centroids[label], inv_cov_matrices[label])
                except KeyError: # couldn't calculate inv covariance of this label, we can also use euclidean instead of mahalanobis!.
                    distance = float("inf")
                if distance < min_distance:
                    min_distance = distance
                    assigned_label = label

            # Assign only if within the calculated threshold for that cluster
            if assigned_label != -1 and min_distance <= thresholds[assigned_label]:
                labels[i] = assigned_label

    return labels
def euclidean_assign_unlabeled_points(data, labels):
    from scipy.spatial.distance import euclidean
    unique_labels = set(labels) - {-1}  # Get unique cluster labels excluding -1

    # Compute cluster centroids
    centroids = {label: np.mean(data[labels == label], axis=0) for label in unique_labels}

    outlier_points = data[labels == -1]
    if len(outlier_points) == 0:  # All points are labeled
        return labels  # Return the same labels

    # Compute distance thresholds based on the 10th percentile of distances
    thresholds = {}
    for label in unique_labels:
        cluster_points = data[labels == label]
        distances = [euclidean(p, centroids[label]) for p in cluster_points]
        thresholds[label] = np.percentile(distances, 10)

    # Assign unlabeled points (-1) to the nearest centroid based on Euclidean distance
    for i, point in enumerate(data):
        if labels[i] == -1:
            min_distance = float('inf')
            assigned_label = -1

            for label in unique_labels:
                distance = euclidean(point, centroids[label])
                if distance < min_distance:
                    min_distance = distance
                    assigned_label = label

            # Assign only if within the calculated threshold for that cluster
            if assigned_label != -1 and min_distance <= thresholds[assigned_label]:
                labels[i] = assigned_label

    return labels
def knn_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k = 25):
    from scipy.stats import mode
    unlabeled_points_mask = current_epoch_labels < 0
    unlabeled_points = train_embedded_data[unlabeled_points_mask,:]
    if len(unlabeled_points) == 0:
        return current_epoch_labels
    knn_for_labelling = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embedded_data)
    indices_for_labelling = knn_for_labelling.kneighbors(unlabeled_points,return_distance=False)
    labels_of_neighbours = current_epoch_labels[indices_for_labelling]
    most_common_labels = mode(labels_of_neighbours, axis = 1, keepdims=False)
    new_labels = most_common_labels.mode
    current_epoch_labels[unlabeled_points_mask] = new_labels
    return current_epoch_labels

def knn_average_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k=25):
    unlabeled_points_mask = current_epoch_labels < 0
    unlabeled_points = train_embedded_data[unlabeled_points_mask, :]
    if len(unlabeled_points) == 0:
        return current_epoch_labels

    # Only fit on labeled data
    labeled_points_mask = current_epoch_labels >= 0
    labeled_data = train_embedded_data[labeled_points_mask, :]
    labeled_labels = current_epoch_labels[labeled_points_mask]

    # kNN fit on labeled data only
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(labeled_data)
    distances, indices = knn.kneighbors(unlabeled_points, return_distance=True)

    new_labels = []

    for i in range(len(unlabeled_points)):
        cluster_distance_sum = defaultdict(list)

        for j in range(k):
            label = labeled_labels[indices[i][j]]
            distance = distances[i][j]
            cluster_distance_sum[label].append(distance)

        # Compute average distance to each cluster among k nearest neighbors
        avg_distances = {label: np.mean(dists) for label, dists in cluster_distance_sum.items()}

        # Assign the label with the smallest average distance
        best_label = min(avg_distances, key=avg_distances.get)
        new_labels.append(best_label)

    current_epoch_labels[unlabeled_points_mask] = new_labels
    return current_epoch_labels
def vote_of_two_knn_methods(train_embedded_data, current_epoch_labels, k=25):
    label1 = knn_average_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k)
    label2 = knn_assign_unlabeled_points(train_embedded_data, current_epoch_labels, k)
    same_mask = label1 == label2
    combined_label = np.copy(label1)
    combined_label[~same_mask] = -1
    return combined_label
#endregion


#region Competitors Initializers
def acedec_fit (n_clusters,
                batch_size,
                device,
                neural_network,
                clustering_n_epochs,
                clustering_optimizer_params,
                data,
                random_state
):

    return ACeDeC(n_clusters=n_clusters,
                   batch_size=batch_size,
                   clustering_optimizer_params=clustering_optimizer_params,
                   neural_network=neural_network,
                   clustering_epochs=clustering_n_epochs,
                   device=device,
                   random_state=random_state
                   ).fit(data)
def dcn_fit(
        n_clusters,
        batch_size,
        device,
        neural_network,
        clustering_n_epochs,
        clustering_optimizer_params,
        data,
        random_state
):
    return DCN(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state
    ).fit(data)
def ddc_fit(
        n_clusters,
        batch_size,
        device,
        neural_network,
        clustering_n_epochs,
        clustering_optimizer_params,
        data,
        random_state
):
    return DDC(
        batch_size=batch_size,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state
    ).fit(data)
def dec_fit(
        n_clusters,
        batch_size,
        device,
        neural_network,
        clustering_n_epochs,
        clustering_optimizer_params,
        data,
        random_state
):
    return DEC(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state
    ).fit(data)
def dipdeck_fit(
        n_clusters,
        batch_size,
        device,
        neural_network,
        clustering_n_epochs,
        clustering_optimizer_params,
        data,
        random_state
):
    return DipDECK (
        n_clusters_init=int(n_clusters * 3),
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        # clustering_epochs=clustering_n_epochs, default is 50.
        device=device,
        random_state=random_state
    ).fit(data)
def dkm_fit(
        n_clusters,
        batch_size,
        device,
        neural_network,
        clustering_n_epochs,
        clustering_optimizer_params,
        data,
        random_state
):
    return DKM(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state
    ).fit(data)
def idec_fit(
        n_clusters,
        batch_size,
        device,
        neural_network,
        clustering_n_epochs,
        clustering_optimizer_params,
        data,
        random_state
):
    return IDEC(
        n_clusters=n_clusters,
        batch_size=batch_size,
        clustering_optimizer_params=clustering_optimizer_params,
        neural_network=neural_network,
        clustering_epochs=clustering_n_epochs,
        device=device,
        random_state=random_state
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
        random_state
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
        random_state
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
        random_state
):
    embedded = _embed_data_np(data, batch_size, neural_network)
    return MeanShift(random_state=random_state).fit(embedded)
#endregion
#region AE
class Autoencoder(torch.nn.Module):
    """A vanilla symmetric autoencoder.

    Args:
        input_dim: size of each input sample
        embedding_size: size of the inner most layer also called embedding

    Attributes:
        encoder: encoder part of the autoencoder, responsible for embedding data points
        decoder: decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    """
    def __init__(self, input_dim:int=2, embedding_size:int=2):
        super(Autoencoder, self).__init__()
        self.fitted = False
        self.work_on_copy = None
        # make a sequential list of all operations you want to apply for encoding a data point
        self.encoder = torch.nn.Sequential(
            # Linear layer (just a matrix multiplication)
            torch.nn.Linear(input_dim, 256),
            # apply an elementwise non-linear function
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(64, embedding_size),
            )

#         self.encoder = torch.nn.Sequential(
#             # Linear layer (just a matrix multiplication)
#             torch.nn.Linear(input_dim, 8*input_dim),
#             # apply an elementwise non-linear function
#             torch.nn.LeakyReLU(inplace=True),
#             torch.nn.Linear(8*input_dim, 4*input_dim),
#             torch.nn.LeakyReLU(inplace=True),
#             torch.nn.Linear(4*input_dim, 2*input_dim),
#             torch.nn.LeakyReLU(inplace=True),
#             torch.nn.Linear(input_dim * 2, embedding_size))

        # make a sequential list of all operations you want to apply for decoding a data point
        # In our case this is a symmetric version of the encoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(256, input_dim),
            )

#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(embedding_size, input_dim * 2),
#             torch.nn.LeakyReLU(inplace=True),
#             torch.nn.Linear(2*input_dim, 4*input_dim),
#             torch.nn.LeakyReLU(inplace=True),
#             torch.nn.Linear(4*input_dim, 8*input_dim),
#             torch.nn.LeakyReLU(inplace=True),
#             torch.nn.Linear(input_dim * 8, input_dim),
#             )

    def encode(self, x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points

        Returns:
            embedded: the embedded data point with dimensionality embedding_size
        """
        return self.encoder(x)

    def decode(self, embedded:torch.Tensor)->torch.Tensor:
        """
        Args:
            embedded: embedded data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(embedded)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """ Applies both encode and decode function.
        The forward function is automatically called if we call self(x).
        Args:
            x: input data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction
#endregion


#region DeepSync
def get_local_core_points(data, k, percent, path):
    if os.path.exists(path):
        core_threshold = None
        return np.load(path).astype(int), core_threshold
    core_points_mask, core_threshold = find_local_core_points_fast(data, k, percent)
    np.save(path, core_points_mask.astype(bool))
    return core_points_mask, core_threshold
def deep_sync_model_gtlabels(device, model, data, dataset_name, trainloader,
                    testloader, optimizer, n_check, k, percent,
                    training_iterations,
                    loss_fn,
                    labels_assignment_method,
                    losses_tracker,
                    eval_tracker,
                    directory,
                    deep_sync_model_path):
    # Create folder to save results in
    imgs_directory = os.path.join(directory, 'images')
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(trackers_path, exist_ok=True)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    labels_over_iterations = np.zeros((len(data), training_iterations + 1)) - 10
    core_points_mask, th = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    original_labels[np.where(np.diag(core_points_mask)==1)[0]] = gt_labels[np.where(np.diag(core_points_mask)==1)[0]]
    labels_over_iterations[:,0] = original_labels
    i = 0
    while(i < training_iterations): 
        for batch, batch_labels, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, dont_propagate, losses_tracker = loss_fn(
                model,
                batch_data,
                iteration_labels,
                losses_tracker,
                device)
            if dont_propagate:
                continue
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        # label assignment
        current_epoch_labels = labels_over_iterations[:,i]
        current_epoch_labels = labels_assignment_method(
            train_embedded_data, current_epoch_labels)
        if i > n_check:
            crop = i
        else :
            crop = None
        high_confidence_labels = check_equality_in_n_consequtive_cols(
            labels_over_iterations,
            n_check,
            crop
        )
        previous_epoch_labels = labels_over_iterations[:,i]
        unified_labels = assign_unified_labels(
            previous_epoch_labels,
            current_epoch_labels
        )
        unified_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]
        # 4 - Store the labels
        labels_over_iterations[:, i+1] = unified_labels
        eval_tracker = update_eval_tracker(gt_labels, unified_labels, eval_tracker)
        saving_path = os.path.join(imgs_directory, '{0:03}'.format(i) + '.jpg')
        plot_2d_dataset(train_embedded_data, unified_labels,
                        centers=None, fixed_scales = False, save=saving_path)
        i += 1
        if np.all(high_confidence_labels):
            print(f"apply early stopping after {i} iterations, all points are labeled confidently.")
            break
    # Create and save the iterations Plots GIF
    create_gif_from_directory(
        imgs_directory,
        os.path.join(directory, f'{dataset_name}.gif'),
        duration=300
    )
    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(),
                        os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(),
                        os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations
def deep_sync_model_ship(device, model, data, dataset_name, trainloader,
                    testloader, optimizer, n_check, k, percent,
                    training_iterations,
                    loss_fn,
                    labels_assignment_method,
                    losses_tracker,
                    eval_tracker,
                    directory,
                    deep_sync_model_path):
    # Create folder to save results in
    imgs_directory = os.path.join(directory, 'images')
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(trackers_path, exist_ok=True)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    labels_over_iterations = np.zeros((len(data), training_iterations + 1)) - 10
    core_points_mask, th = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    core_points = embedded[np.where(np.diag(core_points_mask)==1)[0]]
    ship = SHiP(data=core_points, treeType="DCTree")
    ship_labels = ship.fit_predict(power=2, partitioningMethod="ThreshholdElbow")
    original_labels[np.where(np.diag(core_points_mask)==1)[0]] = torch.FloatTensor(ship_labels)
    eval_tracker = update_eval_tracker(gt_labels, original_labels.numpy(), eval_tracker)

    print("Initial clustering evalution:")
    print("AMI = ", eval_tracker.ami_labeled)
    print("ARI = ", eval_tracker.ari_labeled)
    print(f"labeled points = {eval_tracker.no_labeled_pts}/{len(eval_tracker.predicted_labels)}", )
    print("-------------------------------------")
    
    labels_over_iterations[:,0] = original_labels
    i = 0
    while(i < training_iterations): 
        for batch, batch_labels, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, dont_propagate, losses_tracker = loss_fn(
                model,
                batch_data,
                iteration_labels,
                losses_tracker,
                device)
            if dont_propagate:
                continue
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        # label assignment
        current_epoch_labels = labels_over_iterations[:,i]
        current_epoch_labels = labels_assignment_method(
            train_embedded_data, current_epoch_labels)
        if i > n_check:
            crop = i
        else :
            crop = None
        high_confidence_labels = check_equality_in_n_consequtive_cols(
            labels_over_iterations,
            n_check,
            crop
        )
        previous_epoch_labels = labels_over_iterations[:,i]
        unified_labels = assign_unified_labels(
            previous_epoch_labels,
            current_epoch_labels
        )
        unified_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]
        # 4 - Store the labels
        labels_over_iterations[:, i+1] = unified_labels
        eval_tracker = update_eval_tracker(gt_labels, unified_labels, eval_tracker)
        saving_path = os.path.join(imgs_directory, '{0:03}'.format(i) + '.jpg')
        plot_2d_dataset(train_embedded_data, unified_labels,
                        centers=None, fixed_scales = False, save=saving_path)
        i += 1
        if np.all(high_confidence_labels):
            print(f"apply early stopping after {i} iterations, all points are labeled confidently.")
            break
    # Create and save the iterations Plots GIF
    create_gif_from_directory(
        imgs_directory,
        os.path.join(directory, f'{dataset_name}.gif'),
        duration=300
    )
    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(),
                        os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(),
                        os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations
def deep_sync_model_ship_trueK(device, model, data, dataset_name, trainloader,
                    testloader, optimizer, n_check, k, percent,
                    training_iterations,
                    loss_fn,
                    labels_assignment_method,
                    losses_tracker,
                    eval_tracker,
                    directory,
                    deep_sync_model_path):
    # Create folder to save results in
    imgs_directory = os.path.join(directory, 'images')
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(directory, exist_ok=True)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(trackers_path, exist_ok=True)
    embedded, gt_labels = encode_batchwise(testloader, model, device)
    true_k = len(torch.unique(gt_labels))
    labels_over_iterations = np.zeros((len(data), 1))
    core_points_mask, th = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    core_points = embedded[np.where(np.diag(core_points_mask)==1)[0]]
    ship = SHiP(data=core_points, treeType="DCTree", config={"k":true_k})
    ship_labels = ship.fit_predict(power=2, partitioningMethod="K")
    original_labels[np.where(np.diag(core_points_mask)==1)[0]] = torch.FloatTensor(ship_labels)
    eval_tracker = update_eval_tracker(gt_labels, original_labels.numpy(), eval_tracker)

    print("Initial clustering evalution:")
    print("AMI = ", eval_tracker.ami_labeled)
    print("ARI = ", eval_tracker.ari_labeled)
    print(f"labeled points = {eval_tracker.no_labeled_pts}/{len(eval_tracker.predicted_labels)}", )
    print("-------------------------------------")
    
    labels_over_iterations[:,0] = original_labels
    i = 0
    while(i < training_iterations): 
        for batch, batch_labels, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, dont_propagate, losses_tracker = loss_fn(
                model,
                batch_data,
                iteration_labels,
                losses_tracker,
                device)
            if dont_propagate:
                continue
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        # label assignment
        if i >= n_check:
            crop = i
        else :
            crop = None
        high_confidence_labels = get_high_cof_labels(
            labels_over_iterations,
            n_check,
            crop
        )
        current_epoch_labels = labels_over_iterations[:,i]
        current_epoch_labels[~high_confidence_labels] = -1 # consider points with low confidence as -1 so that they can be assigned to other clusters
        current_epoch_labels = labels_assignment_method(
            train_embedded_data, current_epoch_labels)
        previous_epoch_labels = labels_over_iterations[:,i]
        # unified_labels = assign_unified_labels(
        #     previous_epoch_labels,
        #     current_epoch_labels
        # )
        
        # prevent high confidence labels from changing
        current_epoch_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]
        # prevent labeled points from getting unlabeled
        labeled_to_unlabeled_points_mask = np.logical_and(current_epoch_labels == -1, previous_epoch_labels!=-1)
        current_epoch_labels[labeled_to_unlabeled_points_mask] = previous_epoch_labels[labeled_to_unlabeled_points_mask]
        
        # 4 - Store the labels
        labels_over_iterations = np.hstack((labels_over_iterations, current_epoch_labels.reshape(-1, 1))) # let the labels over iterations matrix grow
        eval_tracker = update_eval_tracker(gt_labels, current_epoch_labels, eval_tracker)
        saving_path = os.path.join(imgs_directory, '{0:03}'.format(i) + '.jpg')
        plot_2d_dataset(train_embedded_data, current_epoch_labels,
                        centers=None, fixed_scales = False, save=saving_path)
        i += 1
        if np.all(high_confidence_labels):
            print(f"apply early stopping after {i} iterations, all points are labeled confidently.")
            break
    # Create and save the iterations Plots GIF
    create_gif_from_directory(
        imgs_directory,
        os.path.join(directory, f'{dataset_name}.gif'),
        duration=300
    )
    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(),
                        os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(),
                        os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations
#endregion