## Learning weakly-supervised metric embeddings 

This repository documents some experiments with siamese/triplet autoencoders, and siamese _variational_ autoencoders. The goal of these experiements was to learn interesting, semantically meaningful low-dimensional representations of the data by: 

1. combining the self-supervised properties of an autoencoder to generate low-dimensional representations of the data 
2. refining these representations using metric learning techniques presented as either triplet or pairwise constraints. 

Pairwise constraints can be formalized as tuples $(A, B)$ where $A$ and $B$ are considered "similar", while triplet constraints are tuples of the form $(A, B, C)$ where $B$ is more similar to $A$ than $C$. 

