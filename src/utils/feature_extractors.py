from abc import ABC, abstractmethod
import numpy as np 
from skimage.feature import hog
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn 
import 

class FeatureExtractor(ABC):
    @abstractmethod
    def process(self):
        raise NotImplementedError 

class HogExtractor(FeatureExtractor):
    def __init__(self, params):
        self.params = params 
    
    def process(self, trainX, trainY):
        print("using HOG as feature extractor.")
        trainX = np.array([hog(dat, multichannel=True, feature_vector=True) for dat in trainX])
        testX = np.array([hog(dat, multichannel=True, feature_vector=True) for dat in testX])
        return trainX, testX

class ResNetExtractor:
    def __init__(self):
        pass 
    
    def process(self, trainX, testX):
        print("using ResNet50 as feature extractor.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = torchvision.models.resnet50(pretrained=True)
        extractor = nn.Sequential(*list(extractor.children())[:-1])
        extractor = extractor.to(device)

        train_feats = []
        test_feats = []
        
        trainloader = DataLoader(trainX, batch_size=128)
        testloader = DataLoader(testX, batch_size=128)
        extractor.eval() 
        with torch.no_grad():
            for data, label in trainloader:
                data = data.to(device)
                feats = extractor(data)
                feats = feats.cpu().reshape(-1, 2048)
                train_feats.append(feats)
            
            for data, label in testloader:
                data = data.to(device)
                feats = extractor(data)
                feats = feats.cpu().reshape(-1, 2048)
                test_feats.append(feats)

        trainX = torch.cat(train_feats, dim=0)
        testX = torch.cat(test_feats, dim=0)

        return trainX, testX 
    