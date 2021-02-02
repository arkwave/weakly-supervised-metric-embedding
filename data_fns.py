import cv2 
import os 
import numpy as np  
import multiprocessing as mp 
from timeit import default_timer as timer

import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision 
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


def read_image(args):
    path, color = args 
    if not color:
        return cv2.imread(path, 0)
    else:
        return cv2.imread(path)

def fetch_and_resize(tags, size=(32, 32), hogify=False):
    filepath = 'drive/My Drive/datasets/'
    full_data = []
    full_labels = []
    color = True if len(size) > 2 else False  
    for tag in tags:
        print('processing %s images' % tag)
        dir = filepath + tag
        all_files = [dir + '/' + file_ for file_ in os.listdir(dir)]
        color_args = [color] * len(all_files)
        input_  = list(zip(all_files, color_args))
        print('reading...', end="")
        pool_ = mp.Pool()
        images = pool_.map(read_image, input_)
        data = [cv2.resize(img, (size[0], size[1])) for img in images]
        if hogify:
            data = [hog(dat, feature_vector=True, multichannel=True) for dat in data]
        print('done.')
        labels = [tag] * len(data)
        full_data.extend(data)
        full_labels.extend(labels)
    full_data = np.array(full_data)
    full_labels = np.array(full_labels) 
    return full_data, full_labels


def add_noise(full_data, noise_factor=5):
    new = full_data + noise_factor * np.random.normal(loc=0., scale=1.0, size=full_data.shape)
    new = new.astype(np.uint8)
    return new 

def load_imagenet(img_size=(32, 32, 3)):
    tags = ['dog', 'cat', 'snake', 'lizard']
    data, labels = fetch_and_resize(tags, size=img_size, hogify=True)
    labels = np.array(labels)
    return process_data(data, labels, composite=False, split_size=0.25)
    return data, labels

def process_data(data, labels, composite=False, split_size=0.25):
    mammals = ['dog', 'cat', 'bovine', 'deer', 'horses']
    reptile = ['snake', 'lizard']
    Y = labels
    X = data 
    trainX, testX, trainLabels, testLabels = train_test_split(X, Y, test_size=split_size, shuffle=True, stratify=Y)

    # composite labels
    if composite:
        print("using composite labels")
        trainY = np.zeros(trainLabels.shape)
        trainY[np.where(np.isin(trainLabels, mammals))[0]] = 1
        trainY[np.where(np.isin(trainLabels, reptile))[0]] = 0

        testY = np.zeros(len(testLabels))
        testY[np.where(np.isin(testLabels, mammals))[0]] = 1
        testY[np.where(np.isin(testLabels, reptile))[0]] = 0 
    else:
        # individual labels
        print("using individual labels")
        trainY = trainLabels
        testY = testLabels
    
    print('trainX shape: ', trainX.shape)
    print('trainY shape: ', trainY.shape)
    print('testX shape: ', testX.shape)
    print('testY shape:', testY.shape)
    
    return trainX, trainY, testX, testY, trainLabels, testLabels

def load_cifar(inds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], feature_extractor='hog', composite=False, 
               composite_labels=None, normalize=False):

    start = timer()
    train_set = CIFAR10('.cifar/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_set = CIFAR10('.cifar/', train=False, download=True, transform=torchvision.transforms.ToTensor())

    trainX = train_set.data
    trainY = np.array(train_set.targets)

    testX = test_set.data
    testY = np.array(test_set.targets)

    print("using only the following indices in CIFAR: ", str(inds))

    train_inds = np.where(np.isin(trainY, inds))[0]
    test_inds = np.where(np.isin(testY, inds))[0]

    trainX = trainX[train_inds, :]
    trainLabels = trainY[train_inds]
    testX = testX[test_inds, :]
    testLabels = testY[test_inds]

    # hogify
    if feature_extractor == 'hog':
        print("using HOG as feature extractor.")
        trainX = np.array([hog(dat, multichannel=True, feature_vector=True) for dat in trainX])
        testX = np.array([hog(dat, multichannel=True, feature_vector=True) for dat in testX])
    
    # resnet
    elif feature_extractor == 'resnet':
        print("using ResNet50 as feature extractor.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = torchvision.models.resnet50(pretrained=True)
        extractor = nn.Sequential(*list(extractor.children())[:-1])
        extractor = extractor.to(device)

        train_feats = []
        test_feats = []
        
        trainloader = DataLoader(train_set, batch_size=128)
        testloader = DataLoader(test_set, batch_size=128)
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

    # composite labels
    if composite:
        print("using the following composite labels: " + str(composite_labels.keys()))
        trainY = np.zeros(trainLabels.shape)
        testY = np.zeros(testLabels.shape)

        for label in composite_labels:
            label_set = composite_labels[label]
            trainY[np.where(np.isin(trainLabels, label_set))] = label 
            testY[np.where(np.isin(testLabels, label_set))] = label 

    else:
        trainY = trainLabels 
        testY = testLabels

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

def load_mnist(normalize=True):

    # get the dataset 
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # grab and normalize training set, apply same parameters to test set.
    train_set = MNIST('./data', train=True, download=True, transform=transforms)
    test_set = MNIST('./data', train=False, download=True, transform=transforms)

    # reshape data
    trainX = train_set.data.reshape(-1, 784)
    trainY = train_set.targets
    testX = test_set.data.reshape(-1, 784)
    testY = test_set.targets

    trainX = trainX.float()
    testX = testX.float()

    if normalize:
        print("normalizing...", end='')
        scaler = StandardScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)
        trainX = torch.from_numpy(trainX)
        testX = torch.from_numpy(testX)

    return trainX.float(), trainY, testX.float(), testY, trainY, testY


def load_data(dataset_name, composite_labels=None, normalize=True, feature_extractor='hog'):
    if dataset_name == 'imagenet':
        return load_imagenet()
    elif dataset_name == 'cifar10':
        if composite_labels is not None:
            return load_cifar(composite=True, composite_labels=composite_labels,
                              feature_extractor=feature_extractor, normalize=normalize)
        else:
            return load_cifar(feature_extractor=feature_extractor, normalize=normalize)
    elif dataset_name.lower() == 'mnist':
        return load_mnist(normalize=normalize) 