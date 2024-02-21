import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from .replay import *

# https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
def add_arrow(line, position=None, direction='right', size=15, color=None, lw=2, alpha=1.0, text=None, text_offset=(0 , 0)):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle='--' if alpha < 1 else '-', alpha=alpha),
        size=size,
    )
    if text is not None:
        line.axes.annotate(text, color=color,
            xytext=(xdata[end_ind] + text_offset[0], ydata[end_ind] + text_offset[1]),
            xy=(xdata[end_ind], ydata[end_ind]),
            size=size,
        )
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def probability_contour(ax, model, device, X, y, threshold, cm=None, cm_bright=None, cbar=True, s=None):
    if cm is None:
        cm = plt.cm.RdBu
    if cm_bright is None:
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    h = .02  # step size in the mesh

    x_min, x_max = -1.1, 1.1
    y_min, y_max = -1.1, 1.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    logits = model(torch.as_tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device))
    logits = logits.detach().cpu().numpy().reshape(xx.shape)

    yhat = sigmoid(logits)

    ax.contour(xx, yy, yhat, levels=[threshold], cmap="Greys", vmin=0, vmax=1)
    contour = ax.contourf(xx, yy, yhat, 25, cmap=cm, alpha=.8, vmin=0, vmax=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k', s=s)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.set_title(r'$\sigma(z) = P(y=1)$')
    ax.grid(False)

    if cbar:
        ax_c = plt.colorbar(contour)
        ax_c.set_ticks([0, .25, .5, .75, 1])
    return ax

def counter_vs_clock(basic_corners=None, basic_colors=None, basic_letters=None, draw_arrows=True, binary=True):
    if basic_corners is None:
        basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        clock_arrows = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
    else:
        clock_arrows = np.array([[0, basic_corners[0][1]], [basic_corners[1][0], 0], 
                                 [0, basic_corners[2][1]], [basic_corners[3][0], 0]])
        
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    if basic_letters is None:
        basic_letters = ['A', 'B', 'C', 'D']        
    
    fig, axs = plt.subplots(1, 1 + draw_arrows, figsize=(3 + 3 * draw_arrows, 3))
    if not draw_arrows:
        axs = [axs]

    corners = basic_corners[:]
    factor = (corners.max(axis=0) - corners.min(axis=0)).max() / 2

    for is_clock in range(1 + draw_arrows):
        if draw_arrows:
            if binary:
                if is_clock:
                    axs[is_clock].text(-.5, 0, 'Clockwise')
                    axs[is_clock].text(-.2, -.25, 'y=1')
                else:
                    axs[is_clock].text(-.5, .0, ' Counter-\nClockwise')
                    axs[is_clock].text(-.2, -.25, 'y=0')

        for i in range(4):
            coords = corners[i]
            color = basic_colors[i]
            letter = basic_letters[i]
            if not binary:
                targets = [2, 3] if is_clock else [1, 2]
            else:
                targets = []
            
            alpha = 0.3 if i in targets else 1.0
            axs[is_clock].scatter(*coords, c=color, s=400, alpha=alpha)

            start = i
            if is_clock:
                end = i + 1 if i < 3 else 0
                arrow_coords = np.stack([corners[start] - clock_arrows[start]*0.15,
                                      corners[end] + clock_arrows[start]*0.15])
            else:
                end = i - 1 if i > 0 else -1
                arrow_coords = np.stack([corners[start] + clock_arrows[end]*0.15,
                                      corners[end] - clock_arrows[end]*0.15])
            alpha = 1.0
            if draw_arrows:
                alpha = 0.3 if ((start in targets) or (end in targets)) else 1.0
            line = axs[is_clock].plot(*arrow_coords.T, c=color, lw=0 if draw_arrows else 2, 
                                      alpha=alpha, linestyle='--' if (alpha < 1) and (not draw_arrows) else '-')[0]
            if draw_arrows:
                add_arrow(line, lw=3, alpha=alpha)

            axs[is_clock].text(*(coords - factor*np.array([.05, 0.05])), letter, c='k' if i in targets else 'w', fontsize=12)
            axs[is_clock].grid(False)
            limits = np.stack([corners.min(axis=0), corners.max(axis=0)])
            limits = limits.mean(axis=0).reshape(2, 1) + 1.2*np.array([[-factor, factor]])
            axs[is_clock].set_xlim(limits[0])
            axs[is_clock].set_ylim(limits[1])
            
            axs[is_clock].set_xlabel(r'$x_0$')
            axs[is_clock].set_ylabel(r'$x_1$', rotation=0)

    fig.tight_layout()
    
    return fig

def plot_sequences(basic_corners=None, basic_colors=None, basic_letters=None, binary=True, target_len=0):
    if basic_corners is None:
        basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    if basic_letters is None:
        basic_letters = ['A', 'B', 'C', 'D']
    
    fig, axs = plt.subplots(4, 2, figsize=(6, 3))

    for d in range(2):
        for b in range(4):
            corners = basic_corners[[(b + i) % 4 for i in range(4)]][slice(None, None, d*2-1)]
            colors = np.array(basic_colors)[[(b + i) % 4 for i in range(4)]][slice(None, None, d*2-1)]
            letters = np.array(basic_letters)[[(b + i) % 4 for i in range(4)]][slice(None, None, d*2-1)]
            for i in range(4):
                axs[b, d].scatter(i, 0, c=colors[i], s=600, alpha=0.3 if (i+target_len)>=4 else 1.0)
                axs[b, d].text(i-.125, -.2, letters[i], c='k' if (i+target_len)>=4 else 'w', fontsize=14)
                axs[b, d].grid(False)
                axs[b, d].set_xticks([])
                axs[b, d].set_yticks([])
                axs[b, d].set_xlim([-.5, 4])
                axs[b, d].set_ylim([-1, 1])
                if binary:
                    axs[b, d].text(4, -.1, f'y={d}')

    fig.tight_layout()
    
    return fig
  
def plot_data(points, directions, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axs = axs.flatten()
    
    for e, ax in enumerate(axs):
        pred_corners = points[e]
        clockwise = directions[e]
        for i in range(4):
            color = 'k'
            ax.scatter(*pred_corners.T, c=color, s=400)
            if i == 3:
                start = -1
            else:
                start = i
            ax.plot(*pred_corners[[start, start+1]].T, c='k', lw=2, alpha=.5, linestyle='-')
            ax.text(*(pred_corners[i] - np.array([.04, 0.04])), str(i+1), c='w', fontsize=12)
            if directions is not None:
                ax.set_title(f'{"Counter-" if not clockwise else ""}Clockwise (y={clockwise})', fontsize=14)

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$", rotation=0)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

    fig.tight_layout()
    return fig

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
    
def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def hidden_states_contour(model, points, directions, cell=False, attr='hidden'):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 12)
    
    hex_list = ['#FF3300', '#FFFFFF', '#000099']
    new_cmap = get_continuous_cmap(hex_list)
    
    device = list(model.parameters())[0].device.type

    for i in range(4):
        ci = i - (2 if i >= 2 else 0)
        ri = (i >= 2)
        ax = fig.add_subplot(gs[ri, ci*5+1:(ci+1)*5+1+(ci==1)])
        model(torch.as_tensor(points).float()[:, :i+1, :].to(device))
        probability_contour(ax, 
                            model.classifier, 
                            device, 
                            getattr(model, attr).detach().cpu().squeeze(), 
                            directions.astype(int), 
                            0.5, 
                            cm=new_cmap, 
                            cm_bright=ListedColormap(['#FF3300', '#000099']), cbar=ci==1)
        ax.set_title(f'Hidden State #{i}')
        ax.set_xlabel(r'$h_0$')
        ax.set_ylabel(r'$h_1$', rotation=0)
        
    fig.tight_layout()

    return fig

def canonical_contour(model, basic_corners=None, basic_colors=None, 
                      cell=False, ax=None, supertitle='', cbar=True, attr='hidden'):
    if basic_corners is None:
        basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    
    corners_clock = []
    corners_anti = []
    color_corners = []

    for b in range(4):
        corners = np.concatenate([basic_corners[b:], basic_corners[:b]])
        corners_clock.append(torch.as_tensor(corners[:]).float())
        corners_anti.append(torch.as_tensor(np.concatenate([corners[:1], corners[1:][::-1]])).float())
        color_corners.append(np.concatenate([basic_colors[b:], basic_colors[:b]])[0])

    points = torch.stack([*corners_clock, *corners_anti])
    
    hex_list = ['#FF3300', '#FFFFFF', '#000099']
    new_cmap = get_continuous_cmap(hex_list)
    
    device = list(model.parameters())[0].device.type

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = ax.get_figure()
    model(points.to(device))
    probability_contour(ax,
                        model.classifier,
                        device, 
                        getattr(model, attr).detach().cpu().squeeze(),
                        [0, 1, 2, 3]*2, 0.5,
                        cm=new_cmap,
                        cm_bright=ListedColormap(['gray', 'g', 'b', 'r']),
                        s=200, cbar=cbar)

    pos = getattr(model, attr).squeeze()[:4].detach().cpu().numpy()
    neg = getattr(model, attr).squeeze()[4:].detach().cpu().numpy()
    for p in pos:
        ax.text(p[0]-.05, p[1]-.03, '+', c='w', fontsize=14, )
    for n in neg:
        ax.text(n[0]-.03, n[1]-.05, '-', c='w', fontsize=14)

    ax.set_title(f'{supertitle}Hidden State #{len(basic_corners)-1}')
    ax.set_xlabel(r'$h_0$')
    ax.set_ylabel(r'$h_1$', rotation=0)
    fig.tight_layout()
    
    return fig

def transformed_inputs(linear_input, basic_corners=None, basic_letters=None, basic_colors=None, ax=None, title=None):
    device = list(linear_input.parameters())[0].device.type
    
    if basic_corners is None:
        basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    if basic_letters is None:
        basic_letters = ['A', 'B', 'C', 'D']
    
    corners = linear_input(torch.as_tensor(basic_corners).float().to(device)).detach().cpu().numpy().squeeze()
    transf_corners = [basic_corners[:], corners]
    factor = (corners.max(axis=0) - corners.min(axis=0)).max() / 2

    ret = False
    if ax is None:
        ret = True
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for corners in transf_corners:
        for i in range(4):
            coords = corners[i]
            color = basic_colors[i]
            letter = basic_letters[i]
            ax.scatter(*coords, c=color, s=400)
            if i == 3:
                start = -1
            else:
                start = i
            ax.plot(*corners[[start, start+1]].T, c='k', lw=1, alpha=.5)
            ax.text(*(coords - factor*np.array([.04, 0.04])), letter, c='w', fontsize=12)
            limits = np.stack([corners.min(axis=0), corners.max(axis=0)])
            limits = limits.mean(axis=0).reshape(2, 1) + 1.2*np.array([[-factor, factor]])
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$", rotation=0)
    
    if ret:
        fig.tight_layout()
        return fig

def build_paths(linear_hidden, linear_input, b=0, basic_corners=None, basic_colors=None):
    device = list(linear_input.parameters())[0].device.type
    
    if basic_corners is None:
        basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    if basic_colors is None:
        basic_colors = ['gray', 'g', 'b', 'r']
    
    corners = np.concatenate([basic_corners[b:], basic_corners[:b]])
    color_corners = np.concatenate([basic_colors[b:], basic_colors[:b]])

    corners_clock = torch.as_tensor(corners[:]).float()
    corners_anti = torch.as_tensor(np.concatenate([corners[:1], corners[1:][::-1]])).float()

    activations = []
    for inputs in [corners_clock, corners_anti]:
        activations.append([])
        hidden = torch.zeros(1, 1, linear_hidden.in_features)
        for i in range(len(inputs)):
            lh = linear_hidden(hidden.to(device))
            li = linear_input(inputs[i].to(device))
            lo = lh + li
            hidden = torch.tanh(lo)
            activations[-1].append({
                'h': lh.squeeze().tolist(),
                'o': lo.squeeze().tolist(),
                'a': hidden.squeeze().tolist()
            })

    path_clock = np.concatenate([np.array([[0, 0]]),
                                 np.array([list(step.values()) for step in activations[0]]).reshape(-1, 2)], axis=0)
    path_anti = np.concatenate([np.array([[0, 0]]), 
                                np.array([list(step.values()) for step in activations[1]]).reshape(-1, 2)], axis=0)
    return path_clock, path_anti, color_corners

def feature_spaces(model, mstates, hstates, gates, titles=None, bounded=None, bounds=(-7.2, 7.2), n_points=4):
    layers = [t[0] for t in list(model.named_modules())[1:]]

    X = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2)
    letters = ['A', 'B', 'C', 'D']
    y = torch.tensor([[0], [1], [2], [3]]).float()

    hidden = torch.zeros(1, 1, 2)

    fig, axs = plt.subplots(n_points, len(layers)+1, figsize=(5*len(layers)+5, 5*n_points))
    axs = np.atleast_2d(axs)

    identity_model = nn.Sequential()
    identity_model.add_module('input', nn.Linear(2, 2))
    with torch.no_grad():
        identity_model.input.weight = nn.Parameter(torch.eye(2))
        identity_model.input.bias = nn.Parameter(torch.zeros(2))
    identity_model

    if titles is None:
        titles = ['hidden'] + layers

    if bounded is None:
        bounded = []

    for i in range(n_points):
        data = build_feature_space(identity_model, 
                                   [identity_model.state_dict()], 
                                   hidden.detach(), 
                                   np.array([i]), 
                                   layer_name='input')
        fs_plot = FeatureSpace(axs[i][0], False, boundary=False).load_data(data)
        _ = FeatureSpace._update(0, fs_plot, colors=['k', 'gray', 'g', 'b', 'r'], s=100)
        axs[i][0].set_title(titles[0])
        if layers[-1] in bounded:
            axs[i][0].set_xlim([-1.05, 1.05])
            axs[i][0].set_ylim([-1.05, 1.05])
        else:
            axs[i][0].set_xlim(bounds)
            axs[i][0].set_ylim(bounds)

        for j, layer in enumerate(layers):
            #c = i + 1
            c = i
            if j >= 0:
                c += 1
            
            data = build_feature_space(model, [mstates[i]], hidden.detach(), np.array([c]), layer_name=layer)
            fs_plot = FeatureSpace(axs[i][j+1], False, boundary=False).load_data(data)
            _ = FeatureSpace._update(0, fs_plot, colors=['k', 'gray', 'g', 'b', 'r'], s=100)
            if layer in gates.keys():
                axs[i][j+1].set_title(titles[j+1][:-1] + '\ [' + ','.join(['{:.2f}'.format(v) for v in gates[layer][i]]) + ']$')
            else:
                axs[i][j+1].set_title(titles[j+1])
                
            if layer in bounded:
                axs[i][j+1].set_xlim([-1.05, 1.05])
                axs[i][j+1].set_ylim([-1.05, 1.05])
            else:
                axs[i][j+1].set_xlim(bounds)
                axs[i][j+1].set_ylim(bounds)

        hidden = model(hidden)

    pad = 5
    for row, ax in enumerate(axs[:, 0]):
        ax.annotate('Input #{}'.format(row), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0),                    
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    if n_points == 1:
        fig.subplots_adjust(left=0.1, top=0.82)
    else:
        fig.subplots_adjust(left=3/(3+(i+2)*5), top=0.95)

    return fig

def paths_clock_and_counter(linear_hidden, linear_input, b=0, basic_letters=None, only_clock=False):
    if only_clock:
        fig, axs_rows = plt.subplots(2, 2, figsize=(10, 10))
    else:
        fig, axs_rows = plt.subplots(2-only_clock, 4, figsize=(20, 10-5*only_clock))
    axs_rows = np.atleast_2d(axs_rows)
    
    names = ['th', 'tx', 'tanh']
    xoff = [.5, -.5, .5, .5, -.5]
    yoff = [-.6, -.6, .5, -.5, -.5]
    titles = ['Input #3', 'Input #2', 'Input #1', 'Input #0', 'Initial']
    if basic_letters is None:
        basic_letters = ['A', 'B', 'C', 'D']

    pad = 5

    path_clock, path_anti, color_corners = build_paths(linear_hidden, linear_input, b=b)

    for row in range(2-only_clock):
        if only_clock:
            axs = axs_rows.flatten()
        else:
            axs = axs_rows[row]
        path = [path_clock, path_anti][row]

        if row:
            letters = ['A', 'D', 'C', 'B']
            colors = ['gray', 'r', 'b', 'g'][::-1]
        else:
            letters = ['A', 'B', 'C', 'D']
            colors = ['gray', 'g', 'b', 'r'][::-1]
        for n, (ax, stop) in enumerate(zip(axs, range(3, -1, -1))):
            ax.set_title(f'{titles[stop]} ({letters[n]})')    
            for i, (s, c) in enumerate(zip([10, 7, 4, 1], colors)):
                if i >= stop:
                    for n, j in enumerate(range(s-1, s+2)):
                        line = ax.plot(*path[j:j+2].T, linewidth=1, marker='o', c=c)[0]
                        add_arrow(line, size=15)
                        if i == stop:
                            ax.text(path[j+1, 0]+xoff[i], path[j+1, 1]+yoff[i], names[n], c=c, fontsize=12)

            if stop == 4:
                ax.text(path[1, 0]+.2, path[1, 1]-.5, names[3], c='k', fontsize=12)
                ax.text(path[0, 0]+.2, path[0, 1]-.5, names[2], c='k', fontsize=1)

            if stop == 0:
                ax.scatter(*path[-1], c=colors[0], zorder=100, marker='*', s=256)

            ax.set_xlabel(r"$h_0$")
            ax.set_ylabel(r"$h_1$", rotation=0)
            ax.set_xlim([-7.2, 7.2])
            ax.set_ylim([-7.2, 7.2])

            # Create a Rectangle patch
            rect = patches.Rectangle((-1,-1),2,2,linewidth=1,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)            
            
    if not only_clock:
        for row, ax in enumerate(axs_rows[:, 0]):
            ax.annotate('Start at {}\n{}Clockwise'.format(basic_letters[b], 'Counter- \n' if row else ''), 
                        xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0),                    
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=3/(3+(i+2)*5), top=0.90)
    return fig

def paths_starts(linear_hidden, linear_input, basic_letters=None):
    fig, axs_rows = plt.subplots(4, 4, figsize=(20, 20))
    names = ['', 'tx', 'tanh', 'th']
    xoff = [.3, -.3, .2, .3, 0]
    yoff = [-.6, -.6, .2, -.2, .2]
    titles = ['Input #3', 'Input #2', 'Input #1', 'Input #0', 'Initial']
    if basic_letters is None:
        basic_letters = ['A', 'B', 'C', 'D']

    pad = 5

    for row in range(4):
        axs = axs_rows[row]

        path, _, color_corners = build_paths(linear_hidden, linear_input, b=row)

        for n, (ax, stop) in enumerate(zip(axs, range(3, -1, -1))):
            letter = chr((ord(basic_letters[row]) + n - 65) % 4 + 65)
            ax.set_title(f'{titles[stop]} ({letter})')
            for i, (s, c) in enumerate(zip([10, 7, 4, 1], color_corners[::-1])):
                if i >= stop:
                    for n, j in enumerate(range(s-1, s+3-(s==10))):
                        line = ax.plot(*path[j:j+2].T, linewidth=1, marker='o', c=c)[0]
                        add_arrow(line, size=15)
                        if i == stop:
                            if n > 0:
                                ax.text(path[j+1, 0]+xoff[i], path[j+1, 1]+yoff[i], names[n], c=c, fontsize=10)

            line = ax.plot(*path[0:2].T, linewidth=1, marker='o', c='k')[0]
            add_arrow(line, size=15)
            if stop == 4:
                ax.text(path[1, 0]+.2, path[1, 1]-.5, names[3], c='k', fontsize=10)
                ax.text(path[0, 0]+.2, path[0, 1]-.5, names[2], c='k', fontsize=10)

            if stop == 0:
                ax.scatter(*path[-1], c=color_corners[-1], zorder=100, marker='*', s=256)

            ax.set_xlabel(r"$h_0$")
            ax.set_ylabel(r"$h_1$", rotation=0)
            ax.set_xlim([-7.2, 7.2])
            ax.set_ylim([-7.2, 7.2])

    for row, ax in enumerate(axs_rows[:, 0]):
        ax.annotate('Start\n at {}'.format(basic_letters[row]), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=3/(3+(i+2)*5), top=0.95)
    return fig

def build_rnn_cell(linear_hidden, activation='tanh'):
    if activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        activation = nn.Sigmoid()
    
    model = nn.Sequential()
    model.add_module('th', linear_hidden)
    model.add_module('addtx', nn.Linear(2, 2))
    model.add_module('activation', activation)
    with torch.no_grad():
        model.addtx.weight = nn.Parameter(torch.eye(2))
        model.addtx.bias = nn.Parameter(torch.zeros(2))
    return model

def add_tx(model, tdata):
    with torch.no_grad():
        model.addtx.bias = nn.Parameter(tdata)
    return model

def generate_rnn_states(linear_hidden, linear_input, X):
    hidden_states = []
    model_states = []
    
    hidden = torch.zeros(1, 1, 2)

    tdata = linear_input(X)
    rcell = build_rnn_cell(linear_hidden)

    for i in range(len(X.squeeze())):
        hidden_states.append(hidden)
        rcell = add_tx(rcell, tdata[:, i, :])
        model_states.append(deepcopy(rcell.state_dict()))
        
        hidden = rcell(hidden)
        
    return rcell, model_states, hidden_states, {}

def disassemble_rnn(rnn_model, layer=''):
    hidden_size = rnn_model.hidden_size
    input_size = rnn_model.input_size
    
    linear_hidden = nn.Linear(hidden_size, hidden_size)
    linear_input = nn.Linear(input_size, hidden_size)

    rnn_state = rnn_model.state_dict()
    
    whh = rnn_state[f'weight_hh{layer}'].to('cpu')
    bhh = rnn_state[f'bias_hh{layer}'].to('cpu')
    
    wih = rnn_state[f'weight_ih{layer}'].to('cpu')
    bih = rnn_state[f'bias_ih{layer}'].to('cpu')

    with torch.no_grad():
        linear_hidden.weight = nn.Parameter(whh)
        linear_hidden.bias = nn.Parameter(bhh)

        linear_input.weight = nn.Parameter(wih)
        linear_input.bias = nn.Parameter(bih)

    return linear_hidden, linear_input

def build_rnn_cell(linear_hidden, activation='tanh'):
    if activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        activation = nn.Sigmoid()
    
    model = nn.Sequential()
    model.add_module('th', linear_hidden)
    model.add_module('addtx', nn.Linear(2, 2))
    model.add_module('activation', activation)
    with torch.no_grad():
        model.addtx.weight = nn.Parameter(torch.eye(2))
        model.addtx.bias = nn.Parameter(torch.zeros(2))
    return model

def add_tx(model, tdata):
    with torch.no_grad():
        model.addtx.bias = nn.Parameter(tdata)
    return model

def generate_rnn_states(linear_hidden, linear_input, X):
    hidden_states = []
    model_states = []
    
    hidden = torch.zeros(1, 1, 2)

    tdata = linear_input(X)
    rcell = build_rnn_cell(linear_hidden)

    for i in range(len(X.squeeze())):
        hidden_states.append(hidden)
        rcell = add_tx(rcell, tdata[:, i, :])
        model_states.append(deepcopy(rcell.state_dict()))
        
        hidden = rcell(hidden)
        
    return rcell, model_states, hidden_states, {}

def build_gru_cell(linear_hidden):
    model = nn.Sequential()
    model.add_module('th', linear_hidden)
    model.add_module('rmult', nn.Linear(2, 2))
    model.add_module('addtx', nn.Linear(2, 2))
    model.add_module('activation', nn.Tanh())
    model.add_module('zmult', nn.Linear(2, 2))
    model.add_module('addh', nn.Linear(2, 2))
    with torch.no_grad():
        model.rmult.weight = nn.Parameter(torch.eye(2))
        model.rmult.bias = nn.Parameter(torch.zeros(2))
        
        model.addtx.weight = nn.Parameter(torch.eye(2))
        model.addtx.bias = nn.Parameter(torch.zeros(2))

        model.zmult.weight = nn.Parameter(torch.eye(2))
        model.zmult.bias = nn.Parameter(torch.zeros(2))

        model.addh.weight = nn.Parameter(torch.eye(2))
        model.addh.bias = nn.Parameter(torch.zeros(2))
    return model

def add_tx(model, tdata):
    with torch.no_grad():
        model.addtx.bias = nn.Parameter(tdata)
    return model

def add_h(model, hidden):
    with torch.no_grad():
        model.addh.bias = nn.Parameter(hidden)
    return model

def rgate(model, r):
    with torch.no_grad():
        model.rmult.weight = nn.Parameter(torch.diag(r.squeeze()))
    return model

def zgate(model, z):
    with torch.no_grad():
        model.zmult.weight = nn.Parameter(torch.diag(z.squeeze()))
    return model

def generate_gru_states(n_linear, r_linear, z_linear, X):
    hidden_states = []
    model_states = []
    rs = []
    zs = []
    
    hidden = torch.zeros(1, 1, 2)

    tdata = n_linear[1](X)
    gcell = build_gru_cell(n_linear[0])
    
    for i in range(len(X.squeeze())):
        hidden_states.append(hidden)
        r = torch.sigmoid(r_linear[0](hidden) + r_linear[1](X[:, i:i+1, :]))
        rs.append(r.squeeze().detach().tolist())
        z = torch.sigmoid(z_linear[0](hidden) + z_linear[1](X[:, i:i+1, :]))
        zs.append((1-z).squeeze().detach().tolist())

        gcell = add_tx(gcell, tdata[:, i, :])
        gcell = rgate(gcell, r)
        gcell = zgate(gcell, 1-z)
        gcell = add_h(gcell, (z*hidden)[:, 0, :])
        model_states.append(deepcopy(gcell.state_dict()))

        hidden = gcell(hidden)

    return gcell, model_states, hidden_states, {'rmult': rs, 'zmult': zs}

def disassemble_gru(gru_model, layer=''):
    hidden_size = gru_model.hidden_size
    input_size = gru_model.input_size
    state = gru_model.state_dict()
    
    Wx = state[f'weight_ih{layer}'].to('cpu')
    bx = state[f'bias_ih{layer}'].to('cpu')
    Wxr, Wxz, Wxn = Wx.split(hidden_size, dim=0)
    bxr, bxz, bxn = bx.split(hidden_size, dim=0)

    Wh = state[f'weight_hh{layer}'].to('cpu')
    bh = state[f'bias_hh{layer}'].to('cpu')
    Whr, Whz, Whn = Wh.split(hidden_size, dim=0)
    bhr, bhz, bhn = bh.split(hidden_size, dim=0)    

    n_linear_hidden = nn.Linear(hidden_size, hidden_size)
    n_linear_input = nn.Linear(input_size, hidden_size)

    r_linear_hidden = nn.Linear(hidden_size, hidden_size)
    r_linear_input = nn.Linear(input_size, hidden_size)

    z_linear_hidden = nn.Linear(hidden_size, hidden_size)
    z_linear_input = nn.Linear(input_size, hidden_size)
        
    with torch.no_grad():
        n_linear_hidden.weight = nn.Parameter(Whn)
        n_linear_hidden.bias = nn.Parameter(bhn)
        n_linear_input.weight = nn.Parameter(Wxn)
        n_linear_input.bias = nn.Parameter(bxn)

        r_linear_hidden.weight = nn.Parameter(Whr)
        r_linear_hidden.bias = nn.Parameter(bhr)
        r_linear_input.weight = nn.Parameter(Wxr)
        r_linear_input.bias = nn.Parameter(bxr)
        
        z_linear_hidden.weight = nn.Parameter(Whz)
        z_linear_hidden.bias = nn.Parameter(bhz)
        z_linear_input.weight = nn.Parameter(Wxz)
        z_linear_input.bias = nn.Parameter(bxz)
    
    return ((n_linear_hidden, n_linear_input),
            (r_linear_hidden, r_linear_input), 
            (z_linear_hidden, z_linear_input))

def figure8(linear_hidden, linear_input, X):
    rcell, mstates, hstates, _ = generate_rnn_states(linear_hidden, linear_input, X.unsqueeze(0))
    titles = [r'$hidden\ state\ (h)$',
              r'$transformed\ state\ (t_h)$',
              r'$adding\ t_x (t_h+t_x)$', 
              r'$activated\ state$' + '\n' + r'$h=tanh(t_h+t_x)$']

    return feature_spaces(rcell, mstates, hstates, {}, titles, bounds=(-1.5, 1.5), n_points=1)

def figure13(rnn):
    square = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2)
    linear_hidden, linear_input = disassemble_rnn(rnn, layer='_l0')
    rcell, mstates, hstates, _ = generate_rnn_states(linear_hidden, linear_input, square)
    return transformed_inputs(linear_input, title='RNN')

def figure16(rnn):
    square = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2)
    linear_hidden, linear_input = disassemble_rnn(rnn, layer='_l0')
    rcell, mstates, hstates, _ = generate_rnn_states(linear_hidden, linear_input, square)

    titles = [r'$hidden\ state\ (h)$',
              r'$transformed\ state\ (t_h)$',
              r'$adding\ t_x (t_h+t_x)$', 
              r'$activated\ state$' + '\n' + r'$h=tanh(t_h+t_x)$']

    return feature_spaces(rcell, mstates, hstates, {}, titles, bounded=['activation'])

def figure17(rnn):
    linear_hidden, linear_input = disassemble_rnn(rnn, layer='_l0')
    return paths_clock_and_counter(linear_hidden, linear_input, only_clock=True)

def figure20(model_rnn, model_gru):
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 11)
    titles = ['RNN', 'GRU']
    models = [model_rnn, model_gru]
    for i in range(2):
        ax = fig.add_subplot(gs[0, i*5+0:(i+1)*5+(i==1)])
        fig = canonical_contour(models[i], ax=ax, supertitle=f'{titles[i]}\n', cbar=i==1)
        
    return fig

def figure22(rnn):
    square = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2)
    n_linear, r_linear, z_linear = disassemble_gru(rnn, layer='_l0')
    gcell, mstates, hstates, gates = generate_gru_states(n_linear, r_linear, z_linear, square)
    gcell(hstates[-1])
    titles = [r'$hidden\ state\ (h)$',
              r'$transformed\ state\ (t_h)$',
              r'$reset\ gate\ (r*t_h)$' + '\n' + r'$r=$',
              r'$adding\ t_x$' + '\n' + r'$r*t_h+t_x$', 
              r'$activated\ state$' + '\n' + r'$n=tanh(r*t_h+t_x)$',
              r'$update\ gate\ (n*(1-z))$' + '\n' + r'$1-z=$',
              r'$adding\ z*h$' + '\n' + r'h=$(1-z)*n+z*h$', 
             ]

    return feature_spaces(gcell, mstates, hstates, gates, titles, bounded=['activation', 'zmult', 'addh'])

def figure25(model_rnn, model_gru, model_lstm):
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 16)
    titles = ['RNN', 'GRU', 'LSTM']
    models = [model_rnn, model_gru, model_lstm]
    for i in range(3):
        ax = fig.add_subplot(gs[0, i*5+0:(i+1)*5+(i==2)])
        fig = canonical_contour(models[i], ax=ax, supertitle=f'{titles[i]}\n', cbar=i==2)
    return fig
