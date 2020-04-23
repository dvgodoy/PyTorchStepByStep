import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.linear_model import LinearRegression
plt.style.use('fivethirtyeight')

def figure1(x_train, y_train, b, w):
    # Generates evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    yhat_range = b + w * x_range

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Dataset
    ax.scatter(x_train, y_train)
    # Predictions
    ax.plot(x_range, yhat_range, label='Model\'s predictions', c='k', linestyle='--')

    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .55), c='k')
    ax.legend(loc=0)
    fig.tight_layout()
    return

def figure2(x_train, y_train, b, w):
    # First data point
    x0, y0 = x_train[0][0], y_train[0][0]
    # Generates evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    yhat_range = b + w * x_range

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Dataset
    ax.scatter(x_train, y_train)
    # First data point
    ax.scatter([x0], [y0], c='r')
    # Predictions
    ax.plot(x_range, yhat_range, label='Model\'s predictions', c='k', linestyle='--')
    # Vertical line showing error between point and prediction
    ax.plot([x0, x0], [b + w * x0, y0 - .03], c='r', linewidth=2, linestyle='--')
    ax.arrow(x0, y0 - .03, 0, .03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    ax.arrow(x0, b + w * x0 + .05, 0, -.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)

    # Annotations
    ax.annotate(r'$error_0$', xy=(.8, 1.5))
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .55), c='k')
    ax.legend(loc=0)
    fig.tight_layout()
    return

def figure3(x_train, y_train, b, w, bs, ws, all_losses):
    # Fits a linear regression to find the actual b and w that minimize the loss
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('b')
    ax.set_ylabel('w')
    ax.set_title('Loss Surface')

    # Loss surface
    CS = ax.contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    ax.clabel(CS, inline=1, fontsize=10)
    # Minimum
    ax.scatter(b_minimum, w_minimum, c='k')
    # Random start
    ax.scatter(b, w, c='k')
    # Annotations
    ax.annotate('Random Start', xy=(-.2, 0.05), c='k')
    ax.annotate('Minimum', xy=(.5, 2.2), c='k')

    fig.tight_layout()
    return

def figure4(x_train, y_train, b, w, bs, ws, all_losses):
    # Fits a linear regression to find the actual b and w that minimize the loss
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
    
   # Looks for the closer indexes for the updated b and w inside their respective ranges
    b_idx = np.argmin(np.abs(bs[0, :] - b))
    w_idx = np.argmin(np.abs(ws[:, 0] - w))

    # Closest values for b and w
    fixedb, fixedw = bs[0, b_idx], ws[w_idx, 0]
    
    w_range = ws[:, 0]
    b_range = bs[0, :]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point
    axs[0].scatter(fixedb, fixedw, c='k')
    # Vertical section
    axs[0].plot([fixedb, fixedb], w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random Start', xy=(fixedb + .1, fixedw + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('w')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: b = {:.2f}'.format(fixedb))
    # Loss
    axs[1].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    fig.tight_layout()
    return