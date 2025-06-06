# DeepSynC: Synchronisation-Based Deep Clustering Algorithm

**DeepSynC** is a novel deep clustering algorithm designed for high-dimensional and complex data, such as images. It addresses key limitations of traditional deep clustering methods—especially those based on k-Means—by introducing a synchronisation-based approach that removes the need to predefine the number of clusters and provides a natural stopping criterion for training.

## Key Features

- No need to predefine the number of clusters
- Supports flexible, non-spherical cluster shapes
- Automatic training stop criterion
- Improved synergy between representation learning and clustering

## How DeepSynC Works

1. **Feature Embedding**  
   Input data is passed through a neural network to learn meaningful low-dimensional representations.

2. **Core Point Identification**  
   Within the embedded space, DeepSynC identifies central data points (core points) that represent the densest regions of potential clusters.

3. **Initial Cluster Assignment**  
   Core points are used as the initial seeds for clusters.

4. **Synchronisation-Based Loss**  
   A novel cluster loss function synchronises the embeddings of similar data points, gradually attracting them toward their nearest core points.

5. **Progressive Assignment**  
   As synchronisation proceeds, non-core points are gradually and automatically assigned to the appropriate clusters based on similarity and proximity in the embedding space.

6. **Automatic Stopping**  
   Training stops when the synchronisation stabilizes, eliminating the need for manually defined epochs or convergence checks.

## Installation
The algorithm was implemented and tested using python==3.12.3 with the following dependencies

- clustpy==0.0.2
- torch==2.5.1
- torchvision==0.20.1

In addition to SHiP framework which can be installed by running
- pip install -i https://test.pypi.org/simple/ SHiP-framework==0.1.1b0

## Running the experiments
In order to reproduce the reported results, run ```main.py``` file.