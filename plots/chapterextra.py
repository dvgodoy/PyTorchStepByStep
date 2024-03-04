import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from matplotlib import animation
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['animation.writer'] = 'ffmpeg'

class Basic(object):
    """Basic plot class, NOT to be instantiated directly.
    """
    def __init__(self, ax):
        self._title = ''
        self._custom_title = ''
        self.n_epochs = 0

        self.ax = ax
        self.ax.clear()
        self.fig = ax.get_figure()

    @property
    def title(self):
        title = self._title
        if not isinstance(title, tuple):
            title = (self._title,)
        title = tuple([' '.join([self._custom_title, t]) for t in title])
        return title

    @property
    def axes(self):
        return (self.ax,)

    def load_data(self, **kwargs):
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        pass

    @staticmethod
    def _update(i, object, epoch_start=0):
        pass

    def set_title(self, title):
        """Prepends a custom title to the plot.
        Parameters
        ----------
        title: String
            Custom title to prepend.
        Returns
        -------
        None
        """
        self._custom_title = title

    def plot(self, epoch):
        """Plots data at a given epoch.
        Parameters
        ----------
        epoch: int
            Epoch to use for the plotting.
        Returns
        -------
        fig: figure
            Figure containing the plot.
        """
        self.__class__._update(epoch, self)
        self.fig.tight_layout()
        return self.fig

    def animate(self, epoch_start=0, epoch_end=-1):
        """Animates plotted data from `epoch_start` to `epoch_end`.
        Parameters
        ----------
        epoch_start: int, optional
            Epoch to start the animation from.
        epoch_end: int, optional
            Epoch to end the animation.
        Returns
        -------
        anim: FuncAnimation
            Animation function for the data.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs

        anim = animation.FuncAnimation(self.fig, self.__class__._update,
                                       fargs=(self, epoch_start),
                                       frames=(epoch_end - epoch_start),
                                       blit=True)
        return anim

class LayerViolins(Basic):
    def __init__(self, ax, title=None):
        super(LayerViolins, self).__init__(ax)
        self.values = None
        self.names = None
        self._title = title

    def load_data(self, layer_violins_data):
        self.values = layer_violins_data.values
        self.names = layer_violins_data.names
        self.palette = dict(zip(self.names, sns.palettes.husl_palette(len(self.names), .7)))
        self.n_epochs = len(self.values)
        self._prepare_plot()
        self._update(0, self)
        return self

    def _prepare_plot(self):
        self.line = self.ax.plot([], [])

    @staticmethod
    def _update(i, lv, epoch_start=0):
        assert len(lv.names) == len(lv.values[i]), "Layer names and values have different lengths!"
        epoch = i + epoch_start

        df = pd.concat([pd.DataFrame(layer_values.ravel(),
                                     columns=[layer_name]).melt(var_name='layers', value_name='values')
                        for layer_name, layer_values in zip(lv.names, lv.values[i])])

        lv.ax.clear()
        sns.violinplot(data=df, x='layers', y='values', ax=lv.ax, cut=0, palette=lv.palette, density_norm='width', linewidth=1.5, hue='layers')
        lv.ax.set_xticklabels(df.layers.unique())
        lv.ax.set_xlabel('Layers')
        if lv._title is not None:
            lv.ax.set_ylabel(lv._title)
        lv.ax.set_ylim([df['values'].min(), df['values'].max()])
        lv.ax.set_title('{} - Epoch: {}'.format(lv.title[0], epoch))

        return lv.line
    
LayerViolinsData = namedtuple('LayerViolinsData', ['names', 'values'])

def build_model(input_dim, n_layers, units, activation, use_bn=False):
    if isinstance(units, list):
        assert len(units) == n_layers
    else:
        units = [units] * n_layers
        
    model = nn.Sequential()
    # Adds first hidden layer with input_dim parameter
    model.add_module('h1', nn.Linear(input_dim, units[0], bias=not use_bn))
    model.add_module('a1', activation())
    if use_bn:
        model.add_module('bn1', nn.BatchNorm1d(units[0], affine=False))
    
    # Adds remaining hidden layers
    for i in range(2, n_layers + 1):
        model.add_module('h{}'.format(i), nn.Linear(units[i-2], units[i-1], bias=not use_bn))
        model.add_module('a{}'.format(i), activation())
        if use_bn:
            model.add_module('bn{}'.format(i), nn.BatchNorm1d(units[i-1], affine=False))

    # Adds output layer
    model.add_module('o', nn.Linear(units[n_layers-1], 1))
    return model

def get_plot_data(train_loader, n_layers=5, hidden_units=100, activation_fn=None, use_bn=False, before=True, model=None):
    import sys
    sys.path.append('..')
    from stepbystep.v3 import StepByStep

    if model is None:
        n_features = train_loader.dataset.tensors[0].shape[1]
        if activation_fn is None:
            activation_fn = nn.ReLU
        model = build_model(n_layers, n_features, hidden_units, activation_fn, use_bn, before)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    n_layers = len(list(filter(lambda c: c[0][0] == 'h', model.named_children())))
    
    sbs = StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader)
    sbs.capture_parameters([f'h{i}' for i in range(1, n_layers + 1)])
    sbs.capture_gradients([f'h{i}' for i in range(1, n_layers + 1)])
    sbs.attach_hooks([f'a{i}' for i in range(1, n_layers + 1)])
    sbs.train(1)
    
    names = [f'h{i}' for i in range(1, n_layers + 1)]

    parameters = [[np.array(sbs._parameters[f'h{i}']['weight']).reshape(-1,) for i in range(1, n_layers + 1)]]
    parms_data = LayerViolinsData(names=names, values=parameters)

    gradients = [[np.array(sbs._gradients[f'h{i}']['weight']).reshape(-1,) for i in range(1, n_layers + 1)]]
    gradients_data = LayerViolinsData(names=names, values=gradients)

    activations = [[np.array(sbs.visualization[f'a{i}']).reshape(-1,) for i in range(1, n_layers + 1)]]
    activations_data = LayerViolinsData(names=names, values=activations)
        
    return parms_data, gradients_data, activations_data

def plot_violins(parms, gradients, activations):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Weights', 'Activations', 'Gradients']
    parms_plot = LayerViolins(axs[0], 'Weights').load_data(parms)
    act_plot = LayerViolins(axs[1], 'Activations').load_data(activations)
    grad_plot = LayerViolins(axs[2], 'Gradients').load_data(gradients)
    axs[0].set_ylim(np.array(axs[0].axes.get_ylim()) * 1.1)
    axs[1].set_ylim(np.array(axs[1].axes.get_ylim()) + np.array([-.2, .2]))
    for i in range(3): axs[i].set_title(titles[i])
    fig.tight_layout()
    return fig

def make_init_fn(config):
    def weights_init(m):
        for c in config.keys():
            if isinstance(m, c):
                try:
                    weight_init_fn = config[c]['w']
                    weight_init_fn(m.weight)
                except KeyError:
                    pass
                
                if m.bias is not None:
                    try:
                        bias_init_fn = config[c]['b']
                        bias_init_fn(m.bias)
                    except KeyError:
                        pass
    return weights_init

def plot_schemes(n_features, n_layers, hidden_units, loader):
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    act_fns = [nn.Sigmoid, nn.Tanh, nn.ReLU]
    winits = [lambda m: nn.init.normal_(m, mean=0.0, std=0.1),
              lambda m: nn.init.xavier_uniform_(m),
              lambda m: nn.init.kaiming_uniform_(m, nonlinearity='relu')]

    for i in range(3):
        model = build_model(n_features, n_layers, hidden_units, act_fns[i], use_bn=False)   

        torch.manual_seed(13)    
        weights_init = make_init_fn({nn.Linear: {'w': winits[i], 'b': nn.init.zeros_}})
        with torch.no_grad():
            model.apply(weights_init)

        parms, gradients, activations = get_plot_data(loader, model=model)
        act_plot = LayerViolins(axs[0, i], 'Activations').load_data(activations)
        grad_plot = LayerViolins(axs[1, i], 'Gradients').load_data(gradients)

    names = [r'$Sigmoid + N(0,\sigma=0.1)$', r'$Tanh + Xavier$', r'$ReLU + Kaiming$']
    for j in range(2):
        ylims = []
        for i in range(3):
            ylims.append(np.array(axs[j, i].axes.get_ylim()))
            axs[0, i].set_title(names[i])
            axs[1, i].set_title('')
            axs[j, i].label_outer()
        for i in range(3):
            axs[j, i].set_ylim([1.1 * np.array(ylims).min(), 1.1 * np.array(ylims).max()])

    for i in range(3):    
        axs[0, i].set_ylim([-1.1, 8])
        axs[1, i].set_ylim([-0.05, 0.05])

    fig.tight_layout()
    return fig

def plot_scheme_bn(n_features, n_layers, hidden_units, loader):
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    winits = [lambda m: nn.init.normal_(m, mean=0.0, std=0.1),
              lambda m: nn.init.kaiming_uniform_(m, nonlinearity='relu'),
              lambda m: nn.init.normal_(m, mean=0.0, std=0.1),]

    for i in range(3):
        model = build_model(n_features, n_layers, hidden_units, nn.ReLU, use_bn=(i==2))

        torch.manual_seed(13)
        weights_init = make_init_fn({nn.Linear: {'w': winits[i], 'b': nn.init.zeros_}})
        with torch.no_grad():
            model.apply(weights_init)

        parms, gradients, activations = get_plot_data(loader, model=model)
        act_plot = LayerViolins(axs[0, i], 'Activations').load_data(activations)
        grad_plot = LayerViolins(axs[1, i], 'Gradients').load_data(gradients)

    names = [r'$ReLU + N(0,\sigma=0.1)$', r'$ReLU + Kaiming$', r'$ReLU + N(0,\sigma=0.1) + BN$']
    for j in range(2):
        ylims = []
        for i in range(3):
            ylims.append(np.array(axs[j, i].axes.get_ylim()))
            axs[0, i].set_title(names[i])
            axs[1, i].set_title('')
            axs[j, i].label_outer()
    for i in range(3):
        axs[j, i].set_ylim([1.1 * np.array(ylims).min(), 1.1 * np.array(ylims).max()])

    for i in range(3):    
        axs[0, i].set_ylim([-0.5, 8])
        axs[1, i].set_ylim([-0.05, 0.05])

    fig.tight_layout()
    return fig

def distributions(X_reg, y_reg):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(X_reg.view(-1,).numpy())
    axs[0].set_xlabel('Feature Values')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Distribution of X')
    axs[1].hist(y_reg.view(-1,).numpy())
    axs[1].set_xlabel('Target Values')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Distribution of y')
    fig.tight_layout()
    return fig

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

def make_line(ax, point):
    point = np.vstack([[0., 0.], np.array(point.squeeze().tolist())])
    line = ax.plot(*point.T, lw=0)[0]
    return line

def compare_grads(grads_before, grads_after):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1.5])
    ax.set_xlabel('Parameter 0')
    ax.set_ylabel('Parameter 1')
    ax.set_title('Gradients')
    add_arrow(make_line(ax, grads_before), lw=2, color='k', text=r'$grad$', 
              size=12, alpha=1.0, text_offset=(-.13, .03))
    add_arrow(make_line(ax, grads_after), lw=2, color='r', text=r'$clipped\ grad$', 
              size=12, alpha=1.0, text_offset=(-.33, .03))
    fig.tight_layout()
    return fig

def gradient_distrib(sbs1, layer1, sbs2, layer2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(np.array(sbs1._gradients[layer1]['weight']).reshape(-1,), bins=np.linspace(-10, 10, 41))
    axs[0].set_ylim([0, 4000])
    axs[0].set_xlabel('Gradients')
    axs[0].set_ylabel('# Updates')
    axs[0].set_title('Using clip_grad_value_')
    axs[1].hist(np.array(sbs2._gradients[layer2]['weight']).reshape(-1,), bins=np.linspace(-10, 10, 41))
    axs[1].set_ylim([0, 4000])
    axs[1].set_xlabel('Gradients')
    axs[1].label_outer()
    axs[1].set_title('Using hooks')
    fig.tight_layout()
    return fig
