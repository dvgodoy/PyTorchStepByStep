import torch
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
plt.style.use('fivethirtyeight')

def odds(prob):
    return prob / (1 - prob)

def log_odds(prob):
    return np.log(odds(prob))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def split_cm(cm):
    # Actual negatives go in the top row, 
    # above the probability line
    actual_negative = cm[0]
    # Predicted negatives go in the first column
    tn = actual_negative[0]
    # Predicted positives go in the second column
    fp = actual_negative[1]

    # Actual positives go in the bottow row, 
    # below the probability line
    actual_positive = cm[1]
    # Predicted negatives go in the first column
    fn = actual_positive[0]
    # Predicted positives go in the second column
    tp = actual_positive[1]
    
    return tn, fp, fn, tp

def tpr_fpr(cm):
    tn, fp, fn, tp = split_cm(cm)
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return tpr, fpr

def precision_recall(cm):
    tn, fp, fn, tp = split_cm(cm)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return precision, recall

def probability_line(ax, y, probs, threshold, shift=0.0, annot=False, colors=None):
    if colors is None:
        colors = ['r', 'b']
    ax.grid(False)
    ax.set_ylim([-.1, .1])
    ax.axes.get_yaxis().set_visible(False)
    ax.plot([0, 1], [0, 0], linewidth=2, c='k', zorder=1)
    ax.plot([0, 0], [-.1, .1], c='k', zorder=1)
    ax.plot([1, 1], [-.1, .1], c='k', zorder=1)

    tn = (y == 0) & (probs < threshold)
    fn = (y == 0) & (probs >= threshold)
    tp = (y == 1) & (probs >= threshold)
    fp = (y == 1) & (probs < threshold)

    ax.plot([threshold, threshold], [-.1, .1], c='k', zorder=1, linestyle='--')
    ax.scatter(probs[tn], np.zeros(tn.sum()) + shift, c=colors[0], s=150, zorder=2, edgecolor=colors[0], linewidth=3)
    ax.scatter(probs[fn], np.zeros(fn.sum()) + shift, c=colors[0], s=150, zorder=2, edgecolor=colors[1], linewidth=3)

    ax.scatter(probs[tp], np.zeros(tp.sum()) - shift, c=colors[1], s=150, zorder=2, edgecolor=colors[1], linewidth=3)
    ax.scatter(probs[fp], np.zeros(fp.sum()) - shift, c=colors[1], s=150, zorder=2, edgecolor=colors[0], linewidth=3)

    ax.set_xlabel(r'$\sigma(z) = P(y=1)$')
    ax.set_title('Threshold = {}'.format(threshold))

    if annot:
        ax.annotate('TN', xy=(.20, .03), c='k', weight='bold', fontsize=20)
        ax.annotate('FN', xy=(.20, -.08), c='k', weight='bold', fontsize=20)
        ax.annotate('FP', xy=(.70, .03), c='k', weight='bold', fontsize=20)
        ax.annotate('TP', xy=(.70, -.08), c='k', weight='bold', fontsize=20)
    return ax

def probability_contour(ax, model, device, X, y, threshold, cm=None, cm_bright=None):
    if cm is None:
        cm = plt.cm.RdBu
    if cm_bright is None:
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # step size in the mesh

    x_min, x_max = -2.25, 2.25
    y_min, y_max = -2.25, 2.25

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    logits = model(torch.as_tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device))
    logits = logits.detach().cpu().numpy().reshape(xx.shape)

    yhat = sigmoid(logits)

    ax.contour(xx, yy, yhat, levels=[threshold], cmap="Greys", vmin=0, vmax=1)
    contour = ax.contourf(xx, yy, yhat, 25, cmap=cm, alpha=.8, vmin=0, vmax=1)
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    #ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_title(r'$\sigma(z) = P(y=1)$')
    ax.grid(False)

    ax_c = plt.colorbar(contour)
    ax_c.set_ticks([0, .25, .5, .75, 1])
    return ax

def eval_curves_from_probs(y, probabilities, threshs, line=False, annot=False):
    cms = [confusion_matrix(y, (probabilities >= threshold)) for threshold in threshs]
    rates = np.array(list(map(tpr_fpr, cms)))
    precrec = np.array(list(map(precision_recall, cms)))
    return eval_curves(rates[:, 1], rates[:, 0], precrec[:, 1], precrec[:, 0], threshs, line=line, annot=annot)    

def eval_curves(fprs, tprs, recalls, precisions, thresholds, thresholds2=None, line=False, annot=False):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    if thresholds2 is None:
        thresholds2 = thresholds[:]

    marker = '.r-' if line else '.r'
    
    axs[0].plot(fprs, tprs, marker, markersize=12, linewidth=2)
    axs[0].set_xlim([-.05, 1.05])
    axs[0].set_ylim([-.05, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('ROC Curve')

    axs[1].plot(recalls, precisions, marker, markersize=12, linewidth=2)
    axs[1].set_xlim([-.05, 1.05])
    axs[1].set_ylim([-.05, 1.05])
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall Curve')
    
    if annot:
        for thresh, fpr, tpr, prec, rec in zip(thresholds, fprs, tprs, precisions, recalls):
            axs[0].annotate(str(thresh), xy=(fpr - .03, tpr - .07))
            
        for thresh, fpr, tpr, prec, rec in zip(thresholds2, fprs, tprs, precisions, recalls):
            axs[1].annotate(str(thresh), xy=(rec - .03, prec - .07))

    fig.tight_layout()
    return fig



def figure1(X_train, y_train, X_val, y_val, cm_bright=None):
    if cm_bright is None:
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)#, edgecolors='k')
    ax[0].set_xlabel(r'$X_1$')
    ax[0].set_ylabel(r'$X_2$')
    ax[0].set_xlim([-2.3, 2.3])
    ax[0].set_ylim([-2.3, 2.3])
    ax[0].set_title('Generated Data - Train')

    ax[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright)#, edgecolors='k')
    ax[1].set_xlabel(r'$X_1$')
    ax[1].set_ylabel(r'$X_2$')
    ax[1].set_xlim([-2.3, 2.3])
    ax[1].set_ylim([-2.3, 2.3])
    ax[1].set_title('Generated Data - Validation')
    fig.tight_layout()
    
    return fig

def figure2(prob1):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    prob = np.linspace(.01, .99, 99)

    for i in [0, 1]:
        ax[i].plot(prob, odds(prob), linewidth=2)
        ax[i].set_xlabel('Probability')
        if i:
            ax[i].set_yscale('log')
            ax[i].set_ylabel('Odds Ratio (log scale)')
            ax[i].set_title('Odds Ratio (log scale)')
        else:
            ax[i].set_ylabel('Odds Ratio')
            ax[i].set_title('Odds Ratio')
        ax[i].scatter([prob1, .5, (1-prob1)], [odds(prob1), odds(.5), odds(1-prob1)], c='r')

    fig.tight_layout()
    
    return fig

def figure3(prob1):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    prob = np.linspace(.01, .99, 99)

    ax[0].plot(prob, log_odds(prob), linewidth=2)
    ax[0].set_xlabel('Probability')
    ax[0].set_ylabel('Log Odds Ratio')
    ax[0].set_title('Log Odds Ratio')
    ax[0].scatter([prob1, .5, (1-prob1)], [log_odds(prob1), log_odds(.5), log_odds(1-prob1)], c='r')

    ax[1].plot(log_odds(prob), prob, linewidth=2)
    ax[1].set_ylabel('Probability')
    ax[1].set_xlabel('Log Odds Ratio')
    ax[1].set_title('Probability')
    ax[1].scatter([log_odds(prob1), log_odds(.5), log_odds(1-prob1)], [prob1, .5, (1-prob1)], c='r')
    fig.tight_layout()

    return fig

def figure4(prob1):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    prob = np.linspace(.01, .99, 99)

    ax.plot(log_odds(prob), prob, linewidth=2, c='r')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Log Odds Ratio')
    ax.set_title('Sigmoid')
    ax.scatter([log_odds(prob1), log_odds(.5), log_odds(1-prob1)], [prob1, .5, (1-prob1)], c='r')
    fig.tight_layout()

    return fig

def figure7(X, y, model, device, cm=None, cm_bright=None):
    if cm is None:
        cm = plt.cm.RdBu
    if cm_bright is None:
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    fig = plt.figure(figsize=(15, 4.5))

    h = .02  # step size in the mesh

    # x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    # y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    
    x_min, x_max = -2.25, 2.25
    y_min, y_max = -2.25, 2.25
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    logits = model(torch.as_tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device))
    logits = logits.detach().cpu().numpy().reshape(xx.shape)

    yhat = sigmoid(logits)

    # 1st plot
    ax = plt.subplot(1, 3, 1)

    contour = ax.contourf(xx, yy, logits, 25, cmap=cm, alpha=.8)
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # Plot the testing points
    #ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_title(r'$z = b + w_1x_1 + w_2x_2$')
    ax.grid(False)
    ax_c = plt.colorbar(contour)
    ax_c.set_label("$z$", rotation=0)

    # 2nd plot
    ax = fig.add_subplot(1, 3, 2, projection='3d')

    surf = ax.plot_surface(xx, yy, yhat, rstride=1, cstride=1, alpha=.5, cmap=cm, linewidth=0, antialiased=True, vmin=0, vmax=1)
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # Plot the testing points
    #ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_title(r'$\sigma(z) = P(y=1)$')

    ax_c = plt.colorbar(surf)
    ax_c.set_ticks([0, .25, .5, .75, 1])
    ax.view_init(30, 220)

    # 3rd plot
    ax = plt.subplot(1, 3, 3)

    ax.contour(xx, yy, yhat, levels=[.5], cmap="Greys", vmin=0, vmax=1)
    contour = ax.contourf(xx, yy, yhat, 25, cmap=cm, alpha=.8, vmin=0, vmax=1)
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # Plot the testing points
    #ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_title(r'$\sigma(z) = P(y=1)$')
    ax.grid(False)

    ax_c = plt.colorbar(contour)
    ax_c.set_ticks([0, .25, .5, .75, 1])

    plt.tight_layout()
    
    return fig

def one_dimension(x, y, colors=None):
    if colors is None:
        colors = ['r', 'b']
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    ax.grid(False)
    ax.set_ylim([-.1, .1])
    ax.axes.get_yaxis().set_visible(False)
    ax.plot([-3, 3], [0, 0], linewidth=2, c='k', zorder=1)
    ax.plot([0, 0], [-.03, .03], c='k', zorder=1)

    ax.scatter(x[y==1], np.zeros_like(x[y==1]), c=colors[1], s=150, zorder=2, linewidth=3)
    ax.scatter(x[y==0], np.zeros_like(x[y==0]), c=colors[0], s=150, zorder=2, linewidth=3)
    ax.set_xlabel(r'$X_1$')
    ax.set_title('One Dimension')
    
    fig.tight_layout()
    
    return fig

def two_dimensions(x, y, colors=None):
    if colors is None:
        colors = ['r', 'b']
    
    x2 = np.concatenate([x.reshape(-1, 1), (x ** 2).reshape(-1, 1)], axis=1)

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(gs[2, 0])

    ax.grid(False)
    ax.set_ylim([-.1, .1])
    ax.axes.get_yaxis().set_visible(False)
    ax.plot([-3, 3], [0, 0], linewidth=2, c='k', zorder=1)
    ax.plot([0, 0], [-.03, .03], c='k', zorder=1)

    ax.scatter(x[y==1], np.zeros_like(x[y==1]), c=colors[1], s=150, zorder=2, linewidth=3)
    ax.scatter(x[y==0], np.zeros_like(x[y==0]), c=colors[0], s=150, zorder=2, linewidth=3)
    ax.set_xlabel(r'$X_1$')
    ax.set_title('One Dimension')

    ax = fig.add_subplot(gs[:, 1])

    ax.scatter(*x2[y==1, :].T, c='b', s=150, zorder=2, linewidth=3)
    ax.scatter(*x2[y==0, :].T, c='r', s=150, zorder=2, linewidth=3)
    ax.plot([-2, 2], [1, 1], 'k--', linewidth=2)
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2=X_1^2$')
    ax.set_title('Two Dimensions')

    fig.tight_layout()
    return fig

def figure9(x, y, model, device, probabilities, threshold, shift=0.0, annot=False, cm=None, cm_bright=None):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[:, 0])
    probability_contour(ax, model, device, x, y, threshold, cm, cm_bright)
    
    if cm_bright is None:
        colors = ['r', 'b']
    else:
        colors = cm_bright.colors

    ax = fig.add_subplot(gs[1, 1:])
    probability_line(ax, y, probabilities, threshold, shift, annot, colors)

    fig.tight_layout()
    return fig

def figure10(y, probabilities, threshold, shift, annot, colors=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    probability_line(ax, y, probabilities, threshold, shift, annot, colors)
    fig.tight_layout()
    return fig

def figure17(y, probabilities, threshs):
    cms = [confusion_matrix(y, (probabilities >= threshold)) for threshold in threshs]
    rates = np.array(list(map(tpr_fpr, cms)))
    precrec = np.array(list(map(precision_recall, cms)))
    precrec = np.nan_to_num(precrec, nan=1.)
    fig = eval_curves(rates[:, 1], rates[:, 0], precrec[:, 1], precrec[:, 0], threshs, line=True, annot=False)
    return fig

def figure19(y, probabilities, threshs=(.4, .5, .57), colors=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    probability_line(axs[0], y, probabilities, threshs[0], 0.0, False, colors)
    probability_line(axs[1], y, probabilities, threshs[1], 0.0, False, colors)
    probability_line(axs[2], y, probabilities, threshs[2], 0.0, False, colors)
    fig.tight_layout()
    return fig
             
def figure20(y):
    fpr_perfect, tpr_perfect, thresholds1_perfect = roc_curve(y, y)
    prec_perfect, rec_perfect, thresholds2_perfect = precision_recall_curve(y, y)
    fig = eval_curves(fpr_perfect, tpr_perfect, rec_perfect, prec_perfect, thresholds1_perfect, thresholds2_perfect, line=True)
    return fig

def figure21(y, probabilities):
    fpr_random, tpr_random, thresholds1_random = roc_curve(y, probabilities)
    prec_random, rec_random, thresholds2_random = precision_recall_curve(y, probabilities)
    fig = eval_curves(fpr_random, tpr_random, rec_random, prec_random, thresholds1_random, thresholds2_random, line=True)
    axs = fig.axes
    axs[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
    axs[1].plot([0, 1], [y.mean(), y.mean()], 'k--', linewidth=2)
    return fig