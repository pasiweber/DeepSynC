from __future__ import annotations

import numpy as np
import torch
import sys

from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._train_utils import get_trained_network
from clustpy.deep._utils import (
    detect_device,
    set_torch_seed,
    squared_euclidean_distance,
    encode_batchwise,
    run_initial_clustering,
)
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Union

from clustpy.utils import plot_with_transformation
import matplotlib.pyplot as plt


class DeepSynC(BaseEstimator, ClusterMixin):
    """
    TODO: Add Description

    Parameters
    ----------
    batch_size : int
        Size of the data batches. (default: 256)
    n_epochs : int
        Number of n_epochs. (default: 50)
    embedding_size : int
        Size of the embedding within the autoencoder. (default: 10)
    autoencoder : torch.nn.Module
        The input autoencoder. If None a new Autoencoder will be created. (default: None)
    optimizer_params : dict
        Parameters of the optimizer for the clustering procedure.
        Can also include the learning rate. (default: {"lr": 1e-3})
    optimizer_class : torch.optim.Optimizer
        The optimizer class. (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction. (default: torch.nn.MSELoss())
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and
        a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used. (default: None)
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution.
        Can also be of type int. (default: None)
    device : torch.device
        If device is None then it will set to cuda if it is available. (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    autoencoder : torch.nn.Module
        The final autoencoder

    Examples
    --------
    >>> from clustpy.data import create_subspace_data
    >>> from clustpy.deep import DeepSynC
    >>> data, labels = create_subspace_data(1500, subspace_features=(3, 50), random_state=1)
    >>> deepsync = DeepSynC()
    >>> labels = deepsync.fit_predict(data)

    References
    ----------
    //TODO: Fill in when published.
    """

    batch_size: int
    autoencoder: Optional[torch.nn.Module]
    pretrain_epochs: int
    pretrain_optimizer_params: dict
    clustering_epochs: int
    clustering_optimizer_params: dict
    embedding_size: int
    optimizer_params: dict
    optimizer_class: torch.optim.Optimizer
    loss_fn: torch.nn.modules.loss._Loss
    custom_dataloaders = Optional[Tuple[Callable, Callable]]
    random_state: np.random.RandomState
    device: torch.device
    show_progress: bool

    n_clusters: Optional[int]
    labels_: np.ndarray

    def __init__(
        self,
        batch_size: int = 256,
        autoencoder: Optional[torch.nn.Module] = None,
        pretrain_epochs: int = 100,
        pretrain_optimizer_params: dict = {"lr": 1e-3},
        clustering_epochs: int = 100,
        clustering_optimizer_params: dict = {"lr": 1e-3},
        embedding_size: int = 10,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        custom_dataloaders: Optional[Tuple[Callable, Callable]] = None,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        device: Optional[torch.device] = None,
        n_clusters: Optional[int] = None,
        show_progress: bool = False,
    ):
        self.batch_size = batch_size
        self.dc_tree = None
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_epochs = clustering_epochs
        self.clustering_optimizer_params = clustering_optimizer_params
        self.embedding_size = embedding_size
        self.autoencoder = autoencoder
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.custom_dataloaders = custom_dataloaders
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)
        self.device = detect_device(device)
        self.n_clusters = n_clusters
        self.show_progress = show_progress

    def fit(self, X, y=None) -> DeepSynC:
        """
        Cluster the input dataset with the DeepSynC algorithm.
        The resulting cluster labels will be stored in the `labels_` attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set.
        y : np.ndarray
            The labels. (can be ignored)

        Returns
        -------
        self : DeepSynC
            This instance of the DeepSynC algorithm.
        """

        # Create Dataloader
        if self.custom_dataloaders is None:
            trainloader = get_dataloader(X, self.batch_size, drop_last=False, shuffle=True)
            testloader = get_dataloader(X, self.batch_size, drop_last=False, shuffle=False)
        else:
            trainloader, testloader = self.custom_dataloaders
            if trainloader.batch_size != self.batch_size:
                self.batch_size = trainloader.batch_size

        # Create and pretrain Autoencoder
        if self.autoencoder is None:
            architecture = [X.shape[1], 256, 128, 64, self.embedding_size]
            self.autoencoder = FeedforwardAutoencoder(architecture)
            self.autoencoder = self.autoencoder.to(self.device)

        if not self.autoencoder.fitted:
            self.autoencoder = get_trained_network(
                trainloader=trainloader,
                n_epochs=self.pretrain_epochs,
                optimizer_params=self.pretrain_optimizer_params,
                optimizer_class=self.optimizer_class,
                device=self.device,
                loss_fn=self.loss_fn,
                embedding_size=self.embedding_size,
                neural_network=self.autoencoder,
                neural_network_class=FeedforwardAutoencoder,
            )
            self.autoencoder.fitted = False

        # Setup DeepSynC Module
        self.deepsync_module = _DeepSynC_Module(
            autoencoder=self.autoencoder,
            n_epochs=self.clustering_epochs,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.clustering_optimizer_params,
            device=self.device,
            show_progress=self.show_progress,
        )
        if not self.autoencoder.fitted:
            print("Start training with clustering loss.")
            self.deepsync_module.fit(
                X,
                trainloader=trainloader,
                loss_fn=self.loss_fn,
                testloader=testloader,
            )
            self.autoencoder.fitted = True

        embedding = encode_batchwise(testloader, self.autoencoder, self.device)

        self.n_clusters, self.labels_, self.cluster_centers_, _ = run_initial_clustering(
            X=embedding,
            n_clusters=self.n_clusters,
            clustering_class=self.cluster_algorithm,
            clustering_params=self.cluster_algorithm_params,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels.
        """
        dataloader = get_dataloader(X, self.batch_size, drop_last=False, shuffle=False)
        embedding = encode_batchwise(dataloader, self.autoencoder, self.device)

        self.n_clusters, self.labels_, self.cluster_centers_, _ = run_initial_clustering(
            X=embedding,
            n_clusters=self.n_clusters,
            clustering_class=self.cluster_algorithm,
            clustering_params=self.cluster_algorithm_params,
            random_state=self.random_state,
        )
        return self.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and then predict the labels.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels.
        """
        return self.fit(X).predict(X)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Embedds the input data with the learned SHADE Autoencoder.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            Embedded input data.
        """
        dataloader = get_dataloader(X, self.batch_size, drop_last=False, shuffle=False)
        embedding = encode_batchwise(dataloader, self.autoencoder, self.device)
        return embedding


class _DeepSynC_Module(_AbstractAutoencoder):
    """
    The DeepSynC Autoencoder.
    """

    autoencoder: torch.nn.Module
    n_epochs: int
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        n_epochs=100,
        min_points: int = 5,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_params: dict = {},
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ):
        super().__init__()

        self.autoencoder = autoencoder
        self.n_epochs = n_epochs
        self.min_points = min_points
        self.optimizer = optimizer_class(list(autoencoder.parameters()), **optimizer_params)
        self.device = detect_device(device)
        self.show_progress = show_progress

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(embedded)

    def fit(
        self,
        X,
        trainloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss._Loss,
        testloader,
    ) -> _DeepSynC_Module:
        """
        Trains DeepSynC and the autoencoder in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Dataloader to be used for training.
        n_epochs : int
            Number of epochs for the clustering procedure.
        optimizer : torch.optim.Optimizer
            The optimizer for training.
        loss_fn : torch.nn.modules.loss._Loss
            Loss function for the reconstruction.
        device : torch.device
            Device to be trained on.

        Returns
        -------
        self : _DeepSynC_Module
            This instance of the _DeepSynC_Module.
        """
        self.train()
        for epoch_i in tqdm(range(self.n_epochs), file=sys.stdout, desc="Epoch", disable=not self.show_progress):
            # loss_rec_sum = []
            # loss_dens_sum = []
            # Update Network
            for batch in trainloader:
                if len(batch[0]) <= self.min_points:
                    continue

                loss_rec, loss_dens = self._loss(X, batch, loss_fn)
                loss = self.degree_of_reconstruction * loss_rec + self.degree_of_density_preservation * loss_dens
                # loss_rec_sum.append(loss_rec.cpu().detach().numpy())
                # loss_dens_sum.append(loss_dens.cpu().detach().numpy())

                ### Backward pass - update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # print(
            #     epoch_i,
            #     np.array(loss_rec_sum).mean(),
            #     np.array(loss_dens_sum).mean(),
            # )
            # emb = encode_batchwise(testloader, self.autoencoder, self.device)
            # plot_with_transformation(emb, scattersize=1, show_plot=False)
            # dc_tree = DCTree(emb)
            # plot_mst(emb, dc_tree)
            plt.show()

        self.autoencoder.eval()
        self.eval()
        return self

    def _loss(
        self,
        X,
        batch: list,
        loss_fn: torch.nn.modules.loss._Loss,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the autoencoder reconstruction + DeepSynC loss.

        Parameters
        ----------
        batch : list
            The minibatch.
        loss_fn : torch.nn.modules.loss._Loss
            Loss function for the reconstruction.
        device : torch.device
            Device to be trained on.

        Returns
        -------loss
        loss : torch.Tensor
            The final SHADE loss.
        """

        # Reconstrucion
        batch_data = batch[1].to(self.device)
        emb_data = self.encode(batch_data)
        reconstructed = self.decode(emb_data)
        loss_rec = loss_fn(reconstructed, batch_data)

        # Density loss
        if self.dc_tree is None:
            # Batch-wise DCTree
            batch_dc_dists = torch.tensor(
                DCTree(X[batch[0]], min_points=self.min_points).dc_distances(),
                device=self.device,
            )
        else:
            # DCTree of all data points X
            if self.use_matrix_dc_distance:
                batch_dc_dists = torch.tensor(self.matrix_dc_distance[np.ix_(batch[0], batch[0])], device=self.device)
            else:
                batch_dc_dists = torch.tensor(self.dc_tree.dc_distances(batch[0], batch[0]), device=self.device)

        batch_eucl_dists = squared_euclidean_distance(emb_data, emb_data)
        loss_dens = (batch_eucl_dists - batch_dc_dists).pow(2).mean()

        return loss_rec, loss_dens
