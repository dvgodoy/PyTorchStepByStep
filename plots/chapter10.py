import numpy as np
import torch
from matplotlib import pyplot as plt

def hist_encoding(encoding):
    encoding = encoding.cpu().detach().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    axs = axs.flatten()
    for i in range(4):
        data_point = encoding[i][0]
        axs[i].hist(data_point, bins=np.linspace(-3, 3, 15), alpha=.5)
        axs[i].set_xlabel(f'Data Point #{i}')
        axs[i].set_ylabel('# of features')
        axs[i].set_title(f'mean={data_point.mean():.4f}\n var={data_point.var():.4f}', fontsize=16)
        axs[i].set_ylim([0, 10])
        axs[i].label_outer()
    fig.tight_layout()    
    return fig

def hist_layer_normed(encoding, normed):
    encoding = encoding.cpu().detach().numpy()
    normed = normed.cpu().detach()
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for i in range(4):
        data_point = encoding[i][0]
        normed_point = normed.detach()[i][0]
        axs[i].hist(data_point, bins=np.linspace(-3, 3, 15), alpha=.5, label='Original')
        axs[i].hist(normed_point.numpy(), bins=np.linspace(-3, 3, 15), alpha=.5, label='Standardized')
        axs[i].set_xlabel(f'Data Point #{i}')
        axs[i].set_ylabel('# of features')
        axs[i].set_title(f'mean={normed.mean().numpy():.4f}\n std={normed.std(unbiased=False).numpy():.4f}', fontsize=16)
        axs[i].legend()
        axs[i].set_ylim([0, 80])
        axs[i].label_outer()
    fig.tight_layout()
    return fig

def plot_patches(patches, kernel_size=3):
    n, p1, p2, v = patches.shape
    fig, axs = plt.subplots(p1, p2, figsize=(3, 3))
    for i in range(p1):
        for j in range(p2):
            axs[i, j].imshow(patches.squeeze()[i, j].view(kernel_size, -1).cpu().detach().numpy(), cmap=plt.cm.gray)
            axs[i, j].grid(False)
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
    return fig

def plot_seq_patches(seq_patches):
    seq_patches = seq_patches.cpu().detach().numpy()
    fig, axs = plt.subplots(1, seq_patches.shape[0], figsize=(3.5, 4))
    for i in range(seq_patches.shape[0]):
        axs[i].imshow(seq_patches[i].reshape(-1, 1), cmap=plt.cm.gray)
        axs[i].grid(False)
        axs[i].set_xticklabels([])
        axs[i].set_xlabel(i)
        axs[i].set_ylabel('Features')
        axs[i].label_outer()
    fig.suptitle('Sequence')
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.9)
    return fig

def plot_seq_patches_transp(seq_patches, add_cls=False, title=None):
    seq_patches = seq_patches.cpu().detach().numpy()
    seq_patches = np.atleast_3d(seq_patches)
    n, l, d = seq_patches.shape
    fig, saxs = plt.subplots(1+seq_patches.shape[1]+add_cls, n, figsize=(n*6, 6), sharex=True)
    
    if title is None:
        title = 'Sequence'
    for seq_n in range(n):
        axs = saxs[:, seq_n]
        if add_cls:
            sub_patches = np.concatenate([np.zeros_like(seq_patches[seq_n, :1]), seq_patches[seq_n]])
        else:
            sub_patches = seq_patches[seq_n]
        axs[0].text(4, 1, f'{title} #{seq_n}', fontsize=16)
        axs[0].grid(False)
        axs[0].set_yticks([])
        for i in range(sub_patches.shape[0]):
            label = i
            if add_cls:
                label = i-1 if i > 0 else '[CLS]'
            axs[i+1].imshow(sub_patches[i].reshape(1, -1), cmap=plt.cm.gray)
            axs[i+1].set_yticklabels([label], rotation=0)
            axs[i+1].grid(False)
            axs[i+1].set_yticks([0])
            #axs[i+1].set_ylabel(i, rotation=0)
            #axs[i+1].label_outer()
        axs[-1].set_xlabel('Features')
    fig.tight_layout()
    return fig

def plot_images(imgs, title=True):
    imgs = imgs.squeeze(1).cpu().detach().numpy()
    imgs = np.atleast_3d(imgs)
    fig, axs = plt.subplots(1, imgs.shape[0], figsize=(6, 3))
    if imgs.shape[0] == 1:
        axs = [axs]
    for i in range(imgs.shape[0]):
        axs[i].imshow(imgs[i], cmap=plt.cm.gray)
        axs[i].grid(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if title:
            axs[i].set_title(f'Image #{i}')
        
    return fig