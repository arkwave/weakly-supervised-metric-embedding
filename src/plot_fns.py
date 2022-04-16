import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
from sklearn.manifold import TSNE


def label_plot(data, embedding, labels, num_images, mu):
    # subsample a section of both dataframes in a stratified fashion. 
    print("data.shape: ", data.shape)

    img_w, img_h = 28, 28
    zoom = 0.5
    inds = [] 
    # for each class, select 200 images per class. 
    for cl in set(labels):
        cl_inds = np.random.choice(np.where(labels == cl)[0],
                                   num_images,
                                   replace=False)
        inds.extend(cl_inds)

    print("total number of points: ", len(inds))
    # filter the data to obtain the images selected for each class. 
    plt_trainX = pd.DataFrame(data[inds, :]).reset_index(drop=True)
    plt_embedding = embedding[inds, :]

    fig, ax = plt.subplots(figsize=(24,16))

    # plot the image at the position corresponding to the embedding. 
    for i, row in plt_trainX.iterrows():
        image = row.values.reshape((img_w, img_h))
        im = OffsetImage(image, zoom=0.4)
        ab = AnnotationBbox(im, (plt_embedding[i, 0], plt_embedding[i, 1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
        ax.update_datalim([(embedding[i, 0], embedding[i, 1])])
        ax.autoscale()
    
    plt.title("Label Plot: mu=%s" % mu)
    plt.show()


def visualize_deformation(embedding_model, input_dim, x=None, y=None):
    if x is None:
        x = np.linspace(-1, 1, 21)
    if y is None:
        y = np.linspace(-1, 1, 21)
    xx, yy = np.meshgrid(x, y)
    all_pts = np.dstack([xx, yy]).reshape(-1, 2)
    colors = []
    for pt in all_pts:
        rgb_val = [pt[0], pt[1], np.linalg.norm([pt[0], pt[1]])]
        colors.append(rgb_val)
    colors = np.array(colors)
    colors = (colors - np.min(colors, axis=0)) / (np.max(colors, axis=0) - np.min(colors, axis=0))

    fig1, ax1 = plt.subplots()
    ax1.set_title('initial color grid')
    ax1.scatter(all_pts[:, 0], all_pts[:, 1], c=colors)

    input_points = all_pts.reshape(-1, 1, 2)
    deformation = embedding_model.predict(input_points)
    deformation = deformation.reshape(deformation.shape[0], deformation.shape[2])

    fig2, ax2 = plt.subplots()
    ax2.set_title('transformed space without noise')
    ax2.scatter(deformation[:, 0], deformation[:, 1], c=colors)

    fig3, ax3 = plt.subplots()
    # add some gaussian noise to the picture, 
    ax3.scatter(deformation[:, 0] + np.random.normal(scale=1e-3, size=len(deformation)),
                deformation[:, 1] + np.random.normal(scale=1e-3, size=len(deformation)),
                c=colors) 
    ax3.set_title('transformed space with noise')

    difference = deformation - all_pts

    fig4, ax4 = plt.subplots() 
    ax4.quiver(all_pts[:, 0], all_pts[:, 1], 
               difference[:, 0], difference[:, 1], 
               np.linalg.norm(difference, axis=1),
                angles='xy')
    ax4.set_title('Deformation Vector Field')
    
    return fig1, fig2, fig3, fig4


def plot_losses(total_loss, metric_loss, recon_loss, num_iters, figsize=(12, 8)):
    
    # smoothing factor = 100
    total_loss = pd.Series(total_loss).rolling(100).mean().dropna()
    metric_loss = pd.Series(metric_loss).rolling(100).mean().dropna()
    recon_loss = pd.Series(recon_loss).rolling(100).mean().dropna()

    num_x = len(total_loss)

    # plot losses    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(range(num_x), total_loss, label='total_loss')
    ax1.legend() 

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(range(num_x), recon_loss, label='recon_loss')
    ax2.legend() 

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(range(num_x), metric_loss, label='metric_loss')
    ax3.legend() 
    plt.show()


def subset_and_plot(train_set, test_set, points_per_class=500):
    """[summary]

    Args:
        train_set ([type]): [description]
        test_set ([type]): [description]
        points_per_class (int, optional): [description]. Defaults to 500.
    """
    # subset the training data to plot TSNE visualization of latent space.
    inds = []
    labels = train_set.targets.unique()
    for label in labels:
        cl_inds = np.random.choice(np.where(labels == label)[0],
                                   points_per_class,
                                   replace=False)
        inds.extend(cl_inds)
    train_data = train_set.data.float().numpy()

    plt_trainX = train_data[inds]
    plt_trainY = train_set.targets.float().numpy()[inds]
    
    tsne = TSNE().fit_transform(plt_trainX)

    plt.figure(figsize=(14,10))
    for label in labels:
        inds = np.where(plt_trainY == label)[0]
        dat = plt_trainX[dat, :]
        plt.scatter(dat[:, 0], dat[:, 1], label=label)
    
    plt.title("t-SNE plot of training set embeddings")
    plt.legend()
    plt.show()


def plot_embeddings(latent_info):
    """
    Plots the obtained latent representation using the params specified in 
    latent_info

    Args:
        latent_info (dict): dictionary with keys ["dim", "mu", "dataset", 
                                                  "train_data", "train_labels",
                                                  "test_data", "test_labels",
                                                  "keys"]
    Returns:
        None. Plots outputs. 

    """
    
    latent_dim = latent_info['dim']
    testLabels = latent_info['test_labels']
    plot_df = latent_info['plot_data']
    keys = latent_info['keys']

    plt.figure(figsize=(10, 10))
    for lab in set(testLabels.flatten()):
        inds = np.where(testLabels == lab)[0]
        dat = plot_df[inds, :]
        label = keys[lab] if keys is not None else lab
        plt.scatter(dat[:, 0], dat[:, 1], label=label, alpha=0.6, marker='.')
    plt.legend()
    plt.axis('equal')
    plt.title("%s Embedding: Mu = %s, latent = %s" % (latent_info['dataset'], latent_info['mu'], latent_dim))
    plt.show()
