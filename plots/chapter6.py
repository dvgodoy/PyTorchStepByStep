import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from PIL import Image
from stepbystep.v2 import StepByStep
from torchvision.transforms import ToPILImage
from sklearn.linear_model import LinearRegression
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CyclicLR, LambdaLR

def EWMA(past_value, current_value, alpha):
    return (1 - alpha) * past_value + alpha * current_value

def calc_ewma(values, period):
    alpha = 2 / (period + 1)
    result = []
    for v in values:
        try:
            prev_value = result[-1]
        except IndexError:
            prev_value = 0

        new_value = EWMA(prev_value, v, alpha)
        result.append(new_value)
    return np.array(result)

def correction(averaged_value, beta, steps):
    return averaged_value / (1 - (beta ** steps))

def figure1(folder='rps'):
    paper = Image.open(f'{folder}/paper/paper02-089.png')
    rock = Image.open(f'{folder}/rock/rock06ck02-100.png')
    scissors = Image.open(f'{folder}/scissors/testscissors02-006.png')

    images = [rock, paper, scissors]
    titles = ['Rock', 'Paper', 'Scissors']

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    for ax, image, title in zip(axs, images, titles):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
    return fig

def calc_corrected_ewma(values, period):
    ewma = calc_ewma(values, period)
    
    alpha = 2 / (period + 1)
    beta = 1 - alpha
    
    result = []
    for step, v in enumerate(ewma):
        adj_value = correction(v, beta, step + 1)
        result.append(adj_value)
        
    return np.array(result)

def figure2(first_images, first_labels):
    fig, axs = plt.subplots(1, 6, figsize=(12, 4))
    titles = ['Paper', 'Rock', 'Scissors']
    for i in range(6):
        image, label = ToPILImage()(first_images[i]), first_labels[i]
        axs[i].imshow(image)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(titles[label], fontsize=12)
    fig.tight_layout()
    return fig

def plot_dist(ax, distrib_outputs, p):
    ax.hist(distrib_outputs, bins=np.linspace(0, 20, 21))
    ax.set_xlabel('Sum of Adjusted Outputs')
    ax.set_ylabel('# of Scenarios')
    ax.set_title('p = {:.2f}'.format(p))
    ax.set_ylim([0, 500])
    mean_value = distrib_outputs.mean()
    ax.plot([mean_value, mean_value], [0, 500], c='r', linestyle='--', label='Mean = {:.2f}'.format(mean_value))
    ax.legend()

def figure7(p, distrib_outputs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_dist(ax, distrib_outputs, p)
    fig.tight_layout()
    return fig

def figure8(ps=(0.1, 0.3, 0.5, 0.9)):
    spaced_points = torch.linspace(.1, 1.1, 11)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, p in zip(axs.flat, ps):
        torch.manual_seed(17)
        distrib_outputs = torch.tensor([F.linear(F.dropout(spaced_points, p=p), 
                                                 weight=torch.ones(11), bias=torch.tensor(0)) 
                                        for _ in range(1000)])    
        plot_dist(ax, distrib_outputs, p)
        ax.label_outer()
    fig.tight_layout()
    return fig

def figure9(first_images, seed=17, p=.33):
    torch.manual_seed(seed)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(ToPILImage()(first_images[0]))
    axs[0].set_title('Original Image')
    axs[0].grid(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(ToPILImage()(F.dropout(first_images[:1], p=p)[0]))
    axs[1].set_title('Regular Dropout')
    axs[1].grid(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(ToPILImage()(F.dropout2d(first_images[:1], p=p)[0]))
    axs[2].set_title('Two-Dimensional Dropout')
    axs[2].grid(False)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    fig.tight_layout()
    return fig

def figure11(losses, val_losses, losses_nodrop, val_losses_nodrop):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(losses, 'b', label='Training Losses - Dropout')
    axs.plot(val_losses, 'r', label='Validation Losses - Dropout')
    axs.plot(losses_nodrop, 'b--', label='Training Losses - No Dropout')
    axs.plot(val_losses_nodrop, 'r--', label='Validation Losses - No Dropout')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Regularizing Effect')
    fig.legend(loc='lower left')
    fig.tight_layout()
    return fig

def figure15(alpha=1/3, periods=5, steps=10):
    t = np.arange(1, steps+1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(t-1, alpha*(1-alpha)**(t-1), label='EWMA')
    ax.bar(t-1, [1/periods]*periods + [0]*(10-periods), color='r', alpha=.3, label='MA')
    ax.set_xticks(t-1)
    ax.grid(False)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Weight')
    ax.set_title(r'$EWMA\ \alpha=\frac{1}{3}$ vs MA (5 periods)')
    ax.legend()
    fig.tight_layout()
    return fig

def ma_vs_ewma(values, periods=19):
    ma19 = pd.Series(values).rolling(min_periods=0, window=periods).mean()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(values, c='k', label='Temperatures')
    ax.plot(ma19, c='k', linestyle='--', label='MA')
    ax.plot(calc_ewma(values, periods), c='r', linestyle='--', label='EWMA')
    ax.plot(calc_corrected_ewma(values, periods), c='r', linestyle='-', label='Bias-corrected EWMA')
    ax.set_title('MA vs EWMA')
    ax.set_ylabel('Temperature')
    ax.set_xlabel('Days')
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig

def figure17(gradients, corrected_gradients, corrected_sq_gradients, adapted_gradients):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.plot(gradients, c='k', label=r'$Gradients$')
    ax.plot(corrected_gradients, c='r', linestyle='-', label=r'$Bias-corrected\ EWMA(grad)$')
    ax.set_title('EWMA for Smoothing')
    ax.set_ylabel('Gradient')
    ax.set_xlabel('Mini-batches')
    ax.set_ylim([-1.5, 1.5])
    ax.legend(fontsize=12)

    ax = axs[1]
    ax.plot(1/(np.sqrt(corrected_sq_gradients)+1e-8), c='b', linestyle='-', label=r'$\frac{1}{\sqrt{Bias-corrected\ EWMA(grad^2)}}$')
    ax.set_title('EWMA for Scaling')
    ax.set_ylabel('Factor')
    ax.set_xlabel('Mini-batches')
    ax.set_ylim([0, 5])
    ax.legend(fontsize=12)

    ax = axs[2]
    ax.plot(gradients, c='k', label='Gradients')
    ax.plot(adapted_gradients, c='g', label='Adapted Gradients')
    ax.set_title('Gradients')
    ax.set_ylabel('Gradient')
    ax.set_xlabel('Mini-batches')
    ax.set_ylim([-1.5, 1.5])
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig

def contour_data(x_tensor, y_tensor):
    linr = LinearRegression()
    linr.fit(x_tensor, y_tensor)
    b, w = linr.intercept_, linr.coef_[0]

    # we have to split the ranges in 100 evenly spaced intervals each
    b_range = np.linspace(.7, 2.3, 101)
    w_range = np.linspace(.7, 2.3, 101)
    # meshgrid is a handy function that generates a grid of b and w
    # values for all combinations
    bs, ws = np.meshgrid(b_range, w_range)
    all_predictions = np.apply_along_axis(
        func1d=lambda x: bs + ws * x, 
        axis=1, 
        arr=x_tensor.numpy()
    )
    all_labels = y_tensor.numpy().reshape(-1, 1, 1)
    all_errors = (all_predictions - all_labels)
    all_losses = (all_errors ** 2).mean(axis=0)
    return b, w, bs, ws, all_losses

def plot_paths(results, b, w, bs, ws, all_losses, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    axs = np.atleast_2d(axs)
    axs = [ax for row in axs for ax in row]
    for i, (ax, desc) in enumerate(zip(axs, results.keys())):
        biases = np.array(results[desc]['parms']['']['linear.bias']).squeeze()
        weights = np.array(results[desc]['parms']['']['linear.weight']).squeeze()
        ax.plot(biases, weights, '-o', linewidth=1, zorder=1, c='k', markersize=4)
        # Loss surface
        CS = ax.contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet, levels=12)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.scatter(b, w, c='r', zorder=2, s=40)
        ax.set_xlim([.7, 2.3])
        ax.set_ylim([.7, 2.3])
        ax.set_xlabel('Bias')
        ax.set_ylabel('Weight')
        ax.set_title(desc)
        ax.label_outer()
    fig = ax.get_figure()
    fig.tight_layout()
    return fig

def plot_losses(results, axs=None):
    n = len(results.keys())
    if axs is None:
        fig, axs = plt.subplots(1, n, figsize=(5*n, 4))
    else:
        fig = axs[0].get_figure()
    for ax, k in zip(axs, results.keys()):
        ax.plot(results[k]['losses'], label='Training Loss', c='b')
        ax.plot(results[k]['val_losses'], label='Validation Loss', c='r')
        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_ylim([1e-3, 1])
        ax.set_title(k)
        ax.legend()
    fig.tight_layout()
    return fig

def momentum(past_value, current_value, beta):
    return beta * past_value + current_value

def calc_momentum(values, beta):
    result = []
    for v in values:
        try:
            prev_value = result[-1]
        except IndexError:
            prev_value = 0

        new_value = momentum(prev_value, v, beta)
        result.append(new_value)
    return np.array(result)

def calc_nesterov(values, beta):
    result = calc_momentum(values, beta)
    return beta * result + values

def figure21(results):
    parm = 'linear.weight'

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in enumerate(axs):
        desc = list(results.keys())[i]
        gradients = np.array(results[desc]['grads'][''][parm]).squeeze()
        momentums = calc_momentum(gradients, 0.9)
        nesterovs = calc_nesterov(gradients, 0.9)
        ax.plot(gradients, c='k', label='Gradients')
        if i > 0:
            ax.plot(momentums, c='r', label='Momentums')
        if i > 1:
            ax.plot(nesterovs, c='b', label='Nesterov Momentums')
        ax.set_title(desc)
        ax.set_ylabel('Gradient')
        ax.set_xlabel('Mini-batches')
        ax.set_ylim([-2, 1.5])
        ax.legend(fontsize=12)

    fig.tight_layout()
    return fig

def plot_scheduler(dummy_optimizer, dummy_scheduler, logscale=True, ax=None):
    learning_rates = []
    for i in range(12):
        current_lr = list(map(lambda d: d['lr'], dummy_scheduler.optimizer.state_dict()['param_groups']))
        learning_rates.append(current_lr)
        dummy_optimizer.step()
        if isinstance(dummy_scheduler, ReduceLROnPlateau):
            dummy_loss = 0.1
            dummy_scheduler.step(dummy_loss)
        else:
            dummy_scheduler.step()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    ax.plot(learning_rates)
    if logscale:
        ax.set_yscale('log')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title(type(dummy_scheduler).__name__)
    fig = ax.get_figure()
    fig.tight_layout()
    return fig

def figure26(dummy_optimizer, dummy_schedulers):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig = plot_scheduler(dummy_optimizer, dummy_schedulers[0], ax=axs[0], logscale=False)
    fig = plot_scheduler(dummy_optimizer, dummy_schedulers[1], ax=axs[1], logscale=False)
    fig = plot_scheduler(dummy_optimizer, dummy_schedulers[2], ax=axs[2], logscale=False)
    axs[0].set_ylim([9e-5, 1e-3])
    axs[1].set_ylim([9e-5, 1e-3])
    axs[2].set_ylim([9e-5, 1e-3])
    axs[0].set_title('CyclicLR - mode=triangular')
    axs[1].set_title('CyclicLR - mode=triangular2')
    axs[2].set_title('CyclicLR - mode=exp_range')
    fig.tight_layout()
    return fig

def compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader=None, schedulers=None, layers_to_hook='', n_epochs=50):
    from stepbystep.v3 import StepByStep
    results = {}
    model_state = deepcopy(model).state_dict()

    for desc, opt in optimizers.items():
        model.load_state_dict(model_state)
        
        optimizer = opt['class'](model.parameters(), **opt['parms'])

        sbs = StepByStep(model, loss_fn, optimizer)
        sbs.set_loaders(train_loader, val_loader)
        
        try:
            if schedulers is not None:
                sched = schedulers[desc]
                scheduler = sched['class'](optimizer, **sched['parms'])
                sbs.set_lr_scheduler(scheduler)
        except KeyError:
            pass        
        
        sbs.capture_parameters(layers_to_hook)
        sbs.capture_gradients(layers_to_hook)
        sbs.train(n_epochs)
        sbs.remove_hooks()

        parms = deepcopy(sbs._parameters)
        grads = deepcopy(sbs._gradients)
        
        lrs = sbs.learning_rates[:]
        if not len(lrs):
            lrs = [list(map(lambda p: p['lr'], optimizer.state_dict()['param_groups']))] * n_epochs

        results.update({desc: {'parms': parms, 
                               'grads': grads,
                               'losses': np.array(sbs.losses),
                               'val_losses': np.array(sbs.val_losses),
                               'state': optimizer.state_dict(), 
                               'lrs': lrs}})
        
    return results

def figure28(results, b, w, bs, ws, all_losses):
    axs = []
    fig = plt.figure(figsize=(15, 12))
    for i in range(3):
        axs.append(plt.subplot2grid((5, 3), (0, i), rowspan=2))
    for i in range(3):
        axs.append(plt.subplot2grid((5, 3), (3, i), rowspan=2))
    for i in range(3):
        axs.append(plt.subplot2grid((5, 3), (2, i)))

    lrs = [results[k]['lrs'] for k in ['SGD + Momentum', 'SGD + Momentum + Step', 'SGD + Momentum + Cycle']]
    for ax, l, title in zip(axs[6:], lrs, ['No Scheduler', 'StepLR', 'CyclicLR']):
        ax.plot(l)
        ax.set_title(title)
        if title == 'CyclicLR':
            ax.set_xlabel('Mini-batches')
        else:
            ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_ylim([0.0, .11])

    fig = plot_paths(results, b, w, bs, ws, all_losses, axs=axs[:6])
    for ax in axs[:6]:
        ax.set_xlabel('Bias')
    fig.tight_layout()
