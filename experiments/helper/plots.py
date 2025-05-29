import os
import torch
import torchvision
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def plot_2d_dataset(data, labels, core_points_mask=None, centers=None, fixed_scales=False, save=None):
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        if centers is not None:
            centers = pca.transform(centers)
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(data[:, 0], data[:, 1], c=labels, label="Data points", alpha=0.6)
    if core_points_mask is not None:
        ax.scatter(
            data[np.where(np.diag(core_points_mask) == 1), 0], data[np.where(np.diag(core_points_mask) == 1), 1], c="r"
        )
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], s=10, c="red")
        for j in range(len(centers)):
            plt.text(centers[j, 0], centers[j, 1], str(j))
    if fixed_scales is True:
        ax.set(xlim=(-10, 10), ylim=(-10, 10))
    else:
        ax.axis("equal")
    if save is not None:
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
    else:
        plt.plot(values, label=legend_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_label is not None:
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
    plt.plot(list1, label=legend_label1, color="b")  # Blue color for the first list
    # Plot the second list
    plt.plot(list2, label=legend_label2, color="r")  # Red color for the second list

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
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], zs=data[:, 2], zdir="z", s=10, c=labels)
    if centers is not None:
        for i in range(len(centers)):
            ax.text(centers[i, 0], centers[i, 1], centers[i, 2], str(i))
    plt.show()


def denormalize(tensor: torch.Tensor, mean: float = 0.1307, std: float = 0.3081) -> torch.Tensor:
    """
    This applies an inverse z-transformation and reshaping to visualize
    the mnist images properly.
    """
    pt_std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    pt_mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    return (tensor.mul(pt_std).add(pt_mean).view(-1, 1, 28, 28) * 255).int().detach()


def plot_images(images: torch.Tensor, pad: int = 0):
    """Aligns multiple images on an N by 10 grid"""

    def imshow(img):
        plt.figure(figsize=(10, 20))
        npimg = img.numpy()
        npimg = np.array(npimg)
        plt.axis("off")
        plt.imshow(np.transpose(npimg, (1, 2, 0)), vmin=0, vmax=1)

    imshow(torchvision.utils.make_grid(images, pad_value=255, normalize=False, padding=pad, nrow=10))
    plt.show()


def print_accuracies(gt_labels, epoch_labels, return_results=True):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari

    labeled_pts_mask = epoch_labels > -1
    lbld_pts_score = ami(gt_labels[labeled_pts_mask], epoch_labels[labeled_pts_mask])
    all_pts_score = ami(gt_labels, epoch_labels)
    lbld_pts_ari = ari(gt_labels[labeled_pts_mask], epoch_labels[labeled_pts_mask])
    all_pts_ari = ari(gt_labels, epoch_labels)
    print("Labeled pts AMI = ", lbld_pts_score)
    print("All pts AMI = ", all_pts_score)
    print("labeled pts ARI = ", lbld_pts_ari)
    print("All pts ARI = ", all_pts_ari)

    if return_results:
        return lbld_pts_score, all_pts_score, lbld_pts_ari, all_pts_ari


def create_gif_from_directory(directory_path, output_filename="output.gif", duration=500):
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
    pattern = r"^\d{3}\.(jpeg|jpg|png)$"
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


def plot_2d_kmeans_result(data, gt_labels, cluster_labels, centers, pairplot=False):
    all_colors = [
        "darkgray",
        "red",
        "peru",
        "yellow",
        "olive",
        "lawngreen",
        "lightsalmon",
        "cyan",
        "darkorange",
        "blue",
        "darkorchid",
        "springgreen",
        "fuchsia",
        "firebrick",
        "crimson",
        "pink",
        "teal",
        "gold",
    ]
    if pairplot:
        import seaborn as sns
        import pandas as pd

        df = pd.DataFrame(data)
        nr_clusters = len(np.unique(cluster_labels))
        columns = df.columns.values
        df["labels"] = cluster_labels
        sns.set(style="ticks")
        colors = all_colors * (nr_clusters // len(all_colors)) + all_colors[: (nr_clusters % len(all_colors))]
        sns.pairplot(df, hue="labels", vars=columns, diag_kind="hist", palette=colors)
    else:
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)
            centers = pca.transform(centers)

        markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"]
        fig, ax = plt.subplots(figsize=(10, 10))
        unique_labels = np.unique(gt_labels)
        for i, l in enumerate(unique_labels):
            ax.scatter(
                data[gt_labels == l, 0],
                data[gt_labels == l, 1],
                c=[all_colors[i % len(all_colors)] for i in cluster_labels[gt_labels == l]],
                s=6,
                marker=markers[i % len(markers)],
            )
        ax.scatter(centers[:, 0], centers[:, 1], c="red")
        for i in range(len(centers)):
            ax.text(centers[i, 0], centers[i, 1], str(i))
        ax.axis("equal")
        # ax.set(xlim=(-2,2), ylim=(-2,2))
    plt.show()


def get_clust_confusion_matrix(gt, pred):
    # cm = confusion_matrix(gt, pred)
    # def _make_cost_m(cm):
    #     s = np.max(cm)
    #     return -cm + s
    # indexes = linear_assignment(_make_cost_m(cm))
    # js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    # cm2 = cm[:, js]
    # return cm2
    gt_clusters = np.unique(gt)
    pred_clusters = np.unique(pred)
    conf_matrix = np.zeros((len(gt_clusters), len(pred_clusters)), dtype=int)
    for i, gt_label in enumerate(gt_clusters):
        point_labels = pred[gt == gt_label]
        labels, cluster_sizes = np.unique(point_labels, return_counts=True)
        for j, pred_label in enumerate(labels):
            conf_matrix[i, np.argwhere(pred_clusters == pred_label)[0][0]] = cluster_sizes[j]
    # Create Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(conf_matrix, cmap="YlGn")
    for i in range(len(gt_clusters)):
        for j in range(len(pred_clusters)):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")
    plt.show()
    return conf_matrix
