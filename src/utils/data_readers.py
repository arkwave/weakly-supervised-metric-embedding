import json
import numpy as np 
from timeit import default_timer as timer
from src.utils.feature_extractors import HogExtractor, ResNetExtractor 
from src.utils.labeling import LabelGenerator 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision.datasets import MNIST, CIFAR10


class DataReader:
    def __init__(self, name, config_path):

        self._datasets = {
            'cifar10': CIFARLoader,
            'mnist': MNISTLoader,
            'imagenet': ImagenetLoader
        }

        self.name = name.lower()
        self.params = json.load(config_path)
    
        self.loader = self._datasets[self.name](self.params)
        self.dataset = self.loader.load_data()
    
    @property 
    def data(self):
        return self.dataset 
 
    def load(self):
        raise NotImplementedError
    
    def process(self):
        raise NotImplementedError


class CIFARLoader(DataReader): 
    
    def __init__(self, params):
        self.params = params  
        self.labeler = LabelGenerator(self.params)

        fe = self.params.get("feature_extractor")

        if fe is not None:
            if fe == 'resnet':
                self.featurizer = ResNetExtractor() 
            elif fe == 'hog':
                self.featurizer = HogExtractor() 
        else:
            self.featurizer = None 

        self.featurizer = ResNetExtractor() if self.params.get("feature_extractor") == 'resnet' else HogExtractor()

    def load(self):
        return self._load(**self.params)

    def _load(self, inds: list=[], feature_extractor: str='hog', 
              normalize: bool=False):
        """_summary_

        Args:
            inds (list, optional): _description_. Defaults to [].
            feature_extractor (str, optional): _description_. Defaults to 'hog'.
            composite (bool, optional): _description_. Defaults to False.
            composite_labels (dict, optional): _description_. Defaults to None.
            normalize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # check to see valid indices 
        indices = list(range(0, 10)) if not inds else inds 

        start = timer()
        train_set = CIFAR10('.cifar/', train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR10('.cifar/', train=False, download=True, transform=torchvision.transforms.ToTensor())

        trainX = train_set.data
        trainY = np.array(train_set.targets)

        testX = test_set.data
        testY = np.array(test_set.targets)

        print("using only the following indices in CIFAR: ", str(indices))

        train_inds = np.where(np.isin(trainY, indices))[0]
        test_inds = np.where(np.isin(testY, indices))[0]

        trainX = trainX[train_inds, :]
        trainLabels = trainY[train_inds]
        testX = testX[test_inds, :]
        testLabels = testY[test_inds]
        
        # perform feature extraction 
        if self.featurizer:
            trainX, testX = self.featurizer.process(trainX, testX)

        # no feature extraction. just flatten and return. 
        else:
            all_dims = np.product(trainX.shape[1:])
            trainX = trainX.reshape(-1, all_dims)
            testX = testX.reshape(-1, all_dims)
    
        # normalize
        if normalize:
            print("normalizing...", end='')
            scaler = StandardScaler()
            trainX = scaler.fit_transform(trainX)
            testX = scaler.transform(testX)
            trainX = torch.from_numpy(trainX)
            testX = torch.from_numpy(testX)

        trainY, testY = self.labeler.fit(trainLabels, testLabels)

        # cast everything to tensors
        trainX = torch.as_tensor(trainX)
        testX = torch.as_tensor(testX)
        trainY = torch.as_tensor(trainY)
        testY = torch.as_tensor(testY)
        trainLabels = torch.as_tensor(trainLabels)
        testLabels = torch.as_tensor(testLabels)

        print("trainX shape: ", trainX.shape)
        print("trainY.shape: ", trainY.shape)
        print("done. elapsed: ", timer() - start)

        return trainX.float(), trainY, testX.float(), testY, trainLabels, testLabels 



class ImagenetLoader(DataReader):
    pass 

class MNISTLoader(DataReader):
    def __init__(self):
        pass 
        
    def load(self):
        pass 

    def process(self):
        pass
