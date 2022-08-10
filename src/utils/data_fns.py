import cv2 
import os 
import json 
import requests
import urllib

import numpy as np  
import multiprocessing as mp 
from bs4 import BeautifulSoup
from timeit import default_timer as timer

 
import torch.optim as optim
import torchvision 



"""
TODO: serious refactor of the data input steps: 
1) Class that fetches the data. 
2) Class that transforms the data - can use something similar to whatever is in finforcast, reusable components 
   that can be chained together.

"""


        
    
        


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




# imagenet downloading helper functions
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image


def get_url(id_, 
            general_url="http://image-net.org/api/text/imagenet.synset.geturls?wnid="):
    page = requests.get(general_url + id_).content
    soup = BeautifulSoup(page, 'html.parser')
    return soup


def soups_from_ids(ids):
    soups = {key: "" for key in ids}
    for id_ in ids:
        soup = get_url(ids[id_])
        soups[id_] = soup
    return soups


def download_images(soups):
    missing = 0
    for key, soup in soups.items():
        img_num = 1
        path = 'datasets/' + key + '/'
        urls = str(soup).split('\r\n')
        print('processing %s images' % key)
        for url in urls:
            if img_num > 150:
                break
            try:
                img = url_to_image(url)
                if img is not None:
                    name = str(key) + '_' + str(img_num) + '.jpg'
                    cv2.imwrite(path + name, img)
                    img_num += 1
            except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
                missing += 1
            except TimeoutError as e:
                continue 
                missing += 1
            except:
                missing += 1
            
    print('missing images: ', missing)

def download_imagenet(ids):
    # ids = {'dog': "n02083346",
#        'cat': "n02120997",
#        'snake': "n01726692" ,
#        'lizard':"n01674464", 
#        'horses': "n02374451",
#        'bovine':"n02402010",
#        'deer': "n02430045"}
    soups = soups_from_ids(ids)
    download_images(soups)