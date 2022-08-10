from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np 
from src.plot_fns import label_plot, plot_losses, plot_embeddings
from src.networks import EmbeddingEncoder, PairwiseEmbeddingEncoder, TripletEmbeddingEncoder, VAE, PairwiseVAE, TripletVAE
from src.utils import PairGenerator, TripletGenerator
from src.losses import ContrastiveLoss, TripletLoss, PairwiseVAELoss, TripletVAELoss, reconstruction_error
from src.training import train_network
import torch 
import torch.optim as optim

from src.utils import DataReader 


def run_knn(trainX, trainY, testX, testY, labels=['dog', 'cat', 'snake', 'lizard'], experiment_name=""):
    print("========= KNN Results: %s ==========" % experiment_name)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(trainX, trainY)   
    train_pred = knn.predict(trainX)
    test_pred = knn.predict(testX)

    train_acc = knn.score(trainX, trainY)
    test_acc = knn.score(testX, testY)

    train_conf_mat = confusion_matrix(trainY, train_pred, labels=labels)
    test_conf_mat = confusion_matrix(testY, test_pred, labels=labels)

    print("Training Accuracy: ", train_acc)
    print("Training Confusion Matrix: \n ", train_conf_mat)
    print("Test Accuracy: ", test_acc)
    print("Test Confusion Matrix: \n ", test_conf_mat)
    print('====================================================')

    return {'train_acc': train_acc, 'test_acc': test_acc, 
            'train_conf_mat': train_conf_mat, 'test_conf_mat': test_conf_mat}


def _experiment(network, latent_dim, datagen, metric_loss, reconstruction_loss,
                trainX, trainY, testX, testY, keys, mu, 
                dataset, trainLabels, testLabels,
                num_iters=20000, batch_size=64, device='cpu'):

    """Overall loop that runs one Embenc experiment. Parameters are as follows:

    Args:
        network ([type]): network being evaluated.
        datagen ([type]): data generator object (either PairGenerator or TripletGenerator)
        metric_loss ([type]): metric loss to be applied to the latent representations.
        reconstruction_loss ([type]): reconstruction loss to be applied. 
        trainX ([type]): training data
        trainY ([type]): training labels
        testX ([type]): testing data
        testY ([type]): test labels
        keys ([type]): mapping between numeric categories in trainY/testY to label names.
        mu ([type]): balancing parameter between reconstruction and metric losses. 
        dataset ([type]): name of the dataset
        trainLabels ([type]): labels associated with trainY, in case composite labels were used for training
        testLabels ([type]): labels associated with testY, in case composite labels were used for training
    
    Returns:
        dict: 
            model: trained model. 
            latent_info: dictionary with keys [train_data, train_label, test_data, test_label, plot_df]
            latent_results: dictionary with output of run_knn on latents. keys are [train_acc, test_acc, 
                            train_conf_mat, test_conf_mat]

    """
    # default optimizer: Adam.
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    # train model
    model, total_loss, recon_loss, metric_loss = train_network(network, datagen, num_iters, metric_loss=metric_loss, 
                                                               reconstruction_loss=reconstruction_loss, 
                                                               optimizer=optimizer, mu=mu, contractive=False,
                                                               device=device)

    # plot losses
    plot_losses(total_loss, metric_loss, recon_loss, num_iters)

    # get predictions
    model.eval()
    model = model.cpu()
    with torch.no_grad():
        latent, reconstructed = model.get_full_pass(testX)
        latent_train, reconstructed_train = model.get_full_pass(trainX)

    # create dictionary for latent data. 
    latent = latent.cpu().numpy() 
    latent_train = latent_train.cpu().numpy()
    testLabels = testLabels.cpu().numpy()
    train_labels = trainLabels.cpu().numpy()

    latent_info = {"dim": latent_dim,
                   "mu": mu,
                   "dataset": dataset,
                   "train_data": latent_train,
                   "train_labels": train_labels,
                   "test_data": latent,
                   "test_labels": testLabels,
                   "keys": keys}

    if latent_dim == 2:
        print('latent = 2, plotting directly.')
        latent_info['plot_data'] = latent 
    else:  
        print("latent dim > 2, fitting tsne...")
        tsne = TSNE(n_components=2, n_iter=3000, n_jobs=-1)
        print(latent.shape)
        tsne_projection = tsne.fit_transform(latent)
        latent_info['plot_data'] = tsne_projection


    # plot reconstruction if mnist. 
    if dataset == 'mnist':
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(testX[0].cpu().numpy().reshape(28, 28))
        ax1.set_title("Original Image")        
        ax2 = fig.add_subplot(1, 2, 2)
        np_arr = reconstructed.cpu().numpy()[0]
        ax2.imshow(np_arr.reshape(28, 28))
        ax2.set_title("reconstruction")
        plt.show()
    
    # plot the embedding 
    plot_embeddings(latent_info)
    
    # knn on the embedding. 
    if torch.is_tensor(trainY):
        if trainY.is_cuda:
            trainY = trainY.cpu()
        trainY = trainY.numpy()
    
    latent_results = run_knn(latent_train, trainY, latent, testY, labels=list(set(testY.flatten())),
                             experiment_name=dataset + ' mu=' + str(mu) + ' latent=' + str(latent_dim))

    if dataset == 'mnist':
        label_plot(testX.cpu().numpy(), latent_info['plot_data'], testY.cpu().numpy(), 200, mu)

    
    results = {"model": model,
               "latent_info": latent_info,
               "latent_results": latent_results}
    
    return results
    

def initialize_network(sim_type, network_type, input_dim, latent_dim, dropout_val=0, device=None):
    """Helper method that grabs the appropriate network type given the parameters. 

    Args:
        sim_type ([string]): "pairwise" or "triplet"
        network_type ([string]): "vae" or "embenc"
        input_dim ([int]): input dimension
        latent_dim ([int]): latent dimension
        dropout_val ([float]): dropout. defaults to 0.
    """
    print("sim type: ", sim_type)
    print("network_type: ", network_type)
    if network_type == 'vae':

        base_vae = VAE(input_dim, latent_dim, dropout=dropout_val, device=device)

        if sim_type == 'pairwise':
            network = PairwiseVAE(base_vae)
        elif sim_type == 'triplet':
            network == TripletVAE(base_vae)

    else:
        base_network = EmbeddingEncoder(input_dim=input_dim, 
                                        latent_dim=latent_dim, 
                                        dropout_val=dropout_val)
        if sim_type == 'pairwise':
            network = PairwiseEmbeddingEncoder(base_network)
        
        if sim_type == 'triplet':
            network = TripletEmbeddingEncoder(base_network)
    
    return network


# def run_experiment(sim_type, network_type, dataset, mus, latent_dims, 
#                    batch_size=64, keys=None, composite_labels=None, 
#                    margin=1, dropout_val=0.2, num_iters=20000,
#                    normalize=True, feature_extractor='hog'):

def run_experiment(reader: DataReader, model_configs: dict) -> dict:
    """Overarching function that runs an experiment. Calls _experiment for each
    combination of mu and latent dim. 

    Args:
        reader: data reader that handles the loading/pre-processing of the dataset specified in data_configs.  
        model_configs: specifies the configuration for the model to be used. 

    Examples for both can be found in configs/data.json and configs/model.json 

    Returns:
        dict: mapping from (mu, latent_dim) to results for _experiment on those params.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("loading data...", end='')

    # select the approprite data
    trainX, trainY, testX, testY, trainLabels, testLabels = reader.load() 

    print("done.")

    # instantiate the generator and the networks based on type. 
    print("initializing data generators and losses...", end='')
    input_dim = trainX.shape[-1]
    if sim_type == 'pairwise':
        datagen = PairGenerator(trainX, trainY, batch_size)
        metric_loss = ContrastiveLoss(margin) if network_type != 'vae' else PairwiseVAELoss(margin)
    
    elif sim_type == 'triplet':
        datagen = TripletGenerator(trainX, trainY, batch_size)
        metric_loss = TripletLoss(margin) if network_type != 'vae' else TripletVAELoss(margin)

    reconstruction_loss = reconstruction_error
    print("done.")

    all_results = {}

    for mu in mus:
        for latent_dim in latent_dims:
                network = initialize_network(sim_type, network_type, input_dim, latent_dim, dropout_val, device=device)

                results = _experiment(network, latent_dim, datagen, metric_loss, reconstruction_loss,
                                      trainX, trainY, testX, testY, keys, mu, 
                                      dataset, trainLabels, testLabels,
                                      num_iters=num_iters, batch_size=batch_size,
                                      device=device)
                
                all_results[(mu, latent_dim)] = results

    return all_results                 