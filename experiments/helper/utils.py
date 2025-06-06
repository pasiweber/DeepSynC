import json
import torch
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_sample(X: torch.Tensor, y: torch.Tensor, sample_size: float, seed: int = 42):
    """
    Perform stratified sampling from PyTorch tensors X and y.

    Args:
        X (torch.Tensor): Feature tensor of shape (n_samples, n_features).
        y (torch.Tensor): Label tensor of shape (n_samples,).
        sample_size (float): Proportion of the dataset to include in the sample (0 < sample_size <= 1).
        seed (int): Random seed for reproducibility.

    Returns:
        X_sampled (torch.Tensor): Stratified sampled features.
        y_sampled (torch.Tensor): Stratified sampled labels.
    """
    if not (0 < sample_size <= 1):
        raise ValueError("sample_size must be between 0 and 1.")
    
    # Convert tensors to numpy for sklearn's stratified splitter
    X_np = X.numpy()
    y_np = y.numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=seed)
    for _, test_index in sss.split(X_np, y_np):
        X_sampled = X[test_index]
        y_sampled = y[test_index]

    return X_sampled, y_sampled


def save_dict_as_json(json_object, file_path):
    with open(file_path, "w") as json_file:
        json.dump(json_object, json_file, indent=4)


def load_json_as_dict(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)
