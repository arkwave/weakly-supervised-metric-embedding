import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.autograd import grad 


# np.random.seed(2)

class TripletGenerator:

    """
    Generates similarity triplets from the dataset provided. 

    Attributes:
        dataset (pandas dataframe): 
        triplets (TYPE): Description
    """
    
    def __init__(self, trainX, trainY, batch_size):
        self.trainX = trainX 
        self.trainY = trainY
        if torch.is_tensor(trainY):
            if trainY.is_cuda:
                self.labels = set(trainY.cpu().numpy().flatten())
            else:
                self.labels = set(trainY.numpy().flatten())
        else:
            self.labels = set(trainY.flatten())

        self.batch_size = batch_size
        self.batch_size = batch_size

    def generate_batch(self):
        """Helper function that generates all valid triplets given a dataset. A triplet
        is a tuple of the form (i, j, k), where i is more similar to j than to k. Similarity is 
        assessed based on class label. 
        
        Returns:
            tuple: anchor, positives, negatives, and batch labels.
        
        """
        indices = []
        for label in self.labels:
            l_indices = np.random.choice(np.where(self.trainY == label)[0], 
                                         self.batch_size // len(self.labels),
                                         replace=False)
            indices.extend(l_indices) 

        indices = np.array(indices)
        batch_labels = self.trainY[indices]
        batch = self.trainX[indices, :]
        anchors, positives, negatives = self.generate_all_triples(batch, batch_labels)

        return (torch.as_tensor(anchors), 
                torch.as_tensor(positives), 
                torch.as_tensor(negatives)), batch_labels

    def generate_all_triples(self, batch, labels):
        """Helper method that generates all triples from a batch. 
        
        Args:
            batch (np array): array of samples from dataset. 
            labels (np array): class labels associated with samples in the batch. 
        
        Returns:
            tuple: anchor, positives, negatives, and batch labels.
        """
        all_triplets = np.array(list(combinations(range(len(labels)), 3))) 
        valid_anchor_pos = all_triplets[labels[all_triplets[:, 0]] == labels[all_triplets[:, 1]]]
        valid_triplets = valid_anchor_pos[labels[valid_anchor_pos[:, 0]] != labels[valid_anchor_pos[:, 2]]]

        anchors = batch[valid_triplets[:, 0]].reshape(-1, batch.shape[1])
        pos = batch[valid_triplets[:, 1]].reshape(-1, batch.shape[1])
        neg = batch[valid_triplets[:, 2]].reshape(-1, batch.shape[1])

        # shuffle all arrays in the same way.  
        p = np.random.permutation(anchors.shape[0])
        anchors = anchors[p, :]
        pos = pos[p, :]
        neg = neg[p, :]

        return anchors, pos, neg


class PairGenerator:

    def __init__(self, trainX, trainY, batch_size):
        self.trainX = trainX 
        self.trainY = trainY  
        if torch.is_tensor(trainY):
            if trainY.is_cuda:
                self.labels = set(trainY.cpu().numpy().flatten())
            else:
                self.labels = set(trainY.numpy().flatten())
        else:
            self.labels = set(trainY.flatten())
        self.batch_size = batch_size

    def generate_batch(self):
        """Randomly samples a batch from the dataset, and returns
        all possible positive and negative pairs from the batch. 
        
        Args:
            batch_size (int): batch from which pairs are generated.
        
        Returns:
            tuple of np arrays: anchors, pairs and labels
        """

        # stratifies the batch selected by class label. 
        indices = []
        for label in self.labels:
            l_indices = np.random.choice(np.where(self.trainY == label)[0], 
                                         self.batch_size//len(self.labels), 
                                         replace=False)
            # print("number of %s: %s" % (label, len(l_indices)))
            indices.extend(l_indices)

        indices = np.array(indices)
        batch_labels = self.trainY[indices]
        batch = self.trainX[indices, :]
        anchors, pairs, labels = self.generate_all_pairs(batch, batch_labels)

        return (torch.as_tensor(anchors), torch.as_tensor(pairs)), torch.as_tensor(labels)

    def generate_all_pairs(self, batch, batch_labels):
        """
        Helper function that generates all possible positive pairs for given batch,
        based on batch_labels. 
        
        Args:
            batch (TYPE): data points in the batch. 
            batch_labels (TYPE): labels corresponding to points in the batch. 
        
        Returns:
            tuple: anchors, pairs and labels (1 if similar, 0 otherwise) 
        """

        # generate all possible pairs 
        all_pairs = np.array(list(combinations(range(len(batch_labels)), 2)))
        
        # select positive and negative pairs. 
        pos_pairs = all_pairs[batch_labels[all_pairs[:, 0]] == batch_labels[all_pairs[:, 1]]]
        neg_pairs = all_pairs[batch_labels[all_pairs[:, 0]] != batch_labels[all_pairs[:, 1]]]
        
        # print("initial size: ", len(batch[0]))
        # print("dims: ", batch.shape)

        end_shape = []
        if len(batch.shape) == 2:
            end_shape = [-1, batch.shape[1]]
        else:
            end_shape = [-1] + list(batch.shape[1:])

        # print('end_shape: ', end_shape)
        # access actual entries from batch corresponding to each. 
        anchor_pos = batch[pos_pairs[:, 0]].reshape(end_shape)
        positives = batch[pos_pairs[:, 1]].reshape(end_shape)
        anchor_neg = batch[neg_pairs[:, 0]].reshape(end_shape)
        negatives = batch[neg_pairs[:, 1]].reshape(end_shape)

        # concatenate.
        anchors = np.vstack([anchor_pos, anchor_neg])
        pairs = np.vstack([positives, negatives])
        
        # generate labels. 1 for similar, 0 for dissimilar. 
        pos_labels = np.ones(shape=(len(positives)))
        neg_labels = np.zeros(shape=(len(negatives)))
        labels = np.hstack([pos_labels, neg_labels])

        # shuffle all arrays in the same way.  
        p = np.random.permutation(anchors.shape[0])
        anchors = anchors[p, :]
        pairs = pairs[p, :]
        labels = labels[p]

        return anchors, pairs, labels


def jacobian(input_, output):
    input_dim = input_.size(-1)
    output_dim = output.size(-1)
    grad_output = torch.ones([1])
    
    outs = []
    inputs = []
    grad_outputs = []
    for i in range(output_dim):
        outs.append(output[i:i+1])
        inputs.append(input_)
        grad_outputs.append(grad_output)
        
    J = torch.zeros(output_dim, input_dim)
    for i in range(output_dim):
        gradient = grad([outs[i]], [inputs[i]], grad_outputs=[grad_outputs[i]],
                        retain_graph=True, create_graph=True, allow_unused=False)
        J[i, :] = gradient[0]
    return J
