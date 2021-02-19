# Learning weakly-supervised metric embeddings 

This repository documents some experiments with siamese/triplet autoencoders, and siamese _variational_ autoencoders. The goal of these experiements was to learn interesting, semantically meaningful low-dimensional representations of the data by: 

1. Combining the self-supervised properties of an autoencoder (variational or otherwise) to generate low-dimensional latent representation of the data
2. Refining the obtained latent representation using contrastive methods on groups of points sampled from the latent space and presented as either pairwise or triplet constraints. 

For reference, pairwise constraints are tuples of the form (A, B) where data points A and B are considered "similar", while triplet constraints are tuples of the form (A, B, C) where B is more similar to A than C. 

## Rationale

Contrastive methods are a popular way to learn semantically meaningful embeddings of high dimensional data. Intuitively, any reasonable embedding should embed similar points close together and dissimilar points far apart - contrastive methods build on this intuition by: 

1. penalizing large distances between points tagged _a-priori_ as similar
2. penalizing small distances (i.e. distances less than some _margin_) between points tagged _a-priori_ as dissimilar.

Suppose we are given two points $x$ and $y$, and a function $f_\theta: \mathbb{R}^n to \mathbb{R}^d$ parameterized by weights $\theta$ and $d <<< n$. Let $z = 1(x = y)$. Our _loss_ between similar vectors is simply the distance between them:

$$\mathcal{L}_{\sim} =  \norm{f_\theta (x) - f_\theta (y)}^2 $$


For dissimilar vectors, we want to minimize the quantity given by $\max(0, l - Formally, we want to find a parameter vector \hat{$\theta$} such that 

$$ \hat{\theta} = \argmin_{\theta} z \cdot \norm{f_\theta (x) - f_\theta (y)}^2 + (1-z)\cdot \max(0, M - \norm{f_\theta (x) - f_\theta (y)})$$  


## Methodology:

The algorithm perform a slightly different (but related) optimization depending on whether pairwise or triplet constraints are used. 

### Optimization for Pairwise Constraints:
Assume that we are using a minibatch of size $K$, with $N$ classes overall. The algorithm then does the following: 

1. Select $K$ points at random from the dataset in a class-stratified manner (i.e. the number of samples per class per minibatch is precisely $\frac{K}{N}$).
2. Generate all possible pairs from this minibatch of size $K$ - this would be a $K \times 2$ vector. Denote the first column $x$, and the second column $y$. 
3. Create a _label vector_ $z$ such that $z_i$ is 1 if $x_i$ and $y_i$ belong to the same class, and 0 otherwise. 
4. Let $d(x, y) = \norm{x - y}_2$. Additionally, let $\mu \in (0, 1)$. Then, our loss function is:

$$\mathcal{L}(x_1, x_2) =  \mu \cdot z \odot d(x, y)^2 + (1-mu) \cdot (1-z) \odot \max(0, l - d(x, y)) $$
