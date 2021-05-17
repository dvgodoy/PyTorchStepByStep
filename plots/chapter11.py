import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def plot_word_vectors(wv, words, other=None):
    vectors = []
    for word in words:
        try:
            vectors.append(wv[word])
        except KeyError:
            if other is not None:
                vectors.append(other[word])
    
    vectors = np.array(vectors)

    fig, axs = plt.subplots(len(words), 1, figsize=(18, len(words)*.7))
    if len(words) == 1:
        axs = [axs]
    
    for i, word in enumerate(words):
        axs[i].imshow(vectors[i].reshape(1, -1), cmap=plt.cm.RdBu, vmin=vectors.min(), vmax=vectors.max())
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels(['', word, ''])
        axs[i].grid(False)
        
    fig.tight_layout()
    return fig

def plot_attention(tokens, alphas):
    n_tokens = max(list(map(len, tokens)))
    batch_size, n_heads, _ = alphas[:, :, 0, :].shape
    alphas = alphas.detach().cpu().numpy()[:, :, 0, :n_tokens]
    fig, axs = plt.subplots(n_heads, batch_size, figsize=(n_tokens * batch_size, n_heads))

    textcolors=["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    for i, axr in enumerate(axs): # row
        for j, ax in enumerate(axr): # col
            data = alphas[j, i]
            im = ax.imshow(np.array(data.tolist()).reshape(1,-1), vmin=0, vmax=1, cmap=plt.cm.gray)
            ax.grid(False) 
            if i == 0:
                ax.set_xticks(np.arange(len(tokens[j])))
                ax.set_xticklabels(tokens[j])
            else:
                ax.set_xticks([])
            ax.set_yticks([-.5, 0, .5], minor=True)
            ax.set_yticklabels(['', f'Head #{i}', ''])
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            for jp in range(data.shape[0]):
                kw.update(color=textcolors[int(im.norm(data[jp]) > .5)])
                text = im.axes.text(jp, 0, valfmt(data[jp], None), **kw)
    return fig