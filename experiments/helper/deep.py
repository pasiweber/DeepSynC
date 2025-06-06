import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from scipy.spatial.distance import cdist
from SHiP import SHiP


def int_to_one_hot(label_tensor, n_labels):
    onehot = torch.zeros([label_tensor.shape[0], n_labels], dtype=torch.float, device=label_tensor.device)
    onehot.scatter_(1, label_tensor.unsqueeze(1).long(), 1.0)
    return onehot

def relabel_negative_ones(labels):
    labels = np.array(labels)  # Ensure it's a NumPy array
    new_labels = labels.copy()
    
    # Find indices where label is -1
    minus_one_indices = np.where(labels == -1)[0]
    
    # Assign each -1 a unique label: -1, -2, -3, -4, ...
    for i, idx in enumerate(minus_one_indices):
        new_labels[idx] = -1 - i  # Start at -1 and go down
    
    return new_labels

def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def encode_batchwise(dataloader, model, device):
    """Utility function for embedding the whole data set in a mini-batch fashion"""
    embeddings = []
    labels = []
    for batch, batch_labels, ids in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0)


def decode_batchwise(dataloader, model, device):
    """Utility function for decoding the whole data set in a mini-batch fashion"""
    decodings = []
    for batch, labels, ids in dataloader:
        batch_data = batch.to(device)
        decodings.append(model(batch_data).detach().cpu())
    return torch.cat(decodings, dim=0).numpy()


def get_train_and_testloader(data, labels, batch_size):
    # create a Dataloader to train the autoencoder in mini-batch fashion
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(data, labels, torch.arange(0, len(labels)))),
        batch_size=batch_size,
        # sample random mini-batches from the data
        shuffle=True,
        drop_last=False,
    )
    # create a Dataloader to test the autoencoder in mini-batch fashion (Important for validation)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(data, labels, torch.arange(0, len(labels)))),
        batch_size=batch_size,
        # Note that we deactivate the shuffling
        shuffle=False,
        drop_last=False,
    )
    return trainloader, testloader


def pretrain_model(
    device,
    pretrained_model_path,
    model,
    data,
    trainloader,
    optimizer,
    loss_fn,
    training_iterations=30,
    verbose=True,
):
    # load model to device
    model.to(device)

    # start training
    i = 0
    # training loop
    while i < training_iterations:
        for batch, _, _ in trainloader:
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


def load_pretrained_model(model, pretrained_model_path, device, work_on_copy=False):
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    if hasattr(model, "fitted"):
        model.fitted = True
    model.work_on_copy = work_on_copy
    return model


# region AE
class Autoencoder(torch.nn.Module):
    """A vanilla symmetric autoencoder.

    Args:
        input_dim: size of each input sample
        embedding_size: size of the inner most layer also called embedding

    Attributes:
        encoder: encoder part of the autoencoder, responsible for embedding data points
        decoder: decoder part of the autoencoder, responsible for reconstructing data points from the embedding
    """

    def __init__(self, input_dim: int = 2, embedding_size: int = 2):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points

        Returns:
            embedded: the embedded data point with dimensionality embedding_size
        """
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedded: embedded data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies both encode and decode function.
        The forward function is automatically called if we call self(x).
        Args:
            x: input data point, can also be a mini-batch of embedded points

        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction


# endregion
