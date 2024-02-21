import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from collections import namedtuple
from copy import deepcopy
from operator import itemgetter
import torch.nn as nn
import torch

def build_2d_grid(xlim, ylim, n_lines=11, n_points=1000):
    """Returns a 2D grid of boundaries given by `xlim` and `ylim`,
     composed of `n_lines` evenly spaced lines of `n_points` each.
    Parameters
    ----------
    xlim : tuple of 2 ints
        Boundaries for the X axis of the grid.
    ylim : tuple of 2 ints
        Boundaries for the Y axis of the grid.
    n_lines : int, optional
        Number of grid lines. Default is 11.
        If n_lines equals n_points, the grid can be used as
        coordinates for the surface of a contourplot.
    n_points: int, optional
        Number of points in each grid line. Default is 1,000.
    Returns
    -------
    lines : ndarray
        For the cases where n_lines is less than n_points, it
        returns an array of shape (2 * n_lines, n_points, 2)
        containing both vertical and horizontal lines of the grid.
        If n_lines equals n_points, it returns an array of shape
        (n_points, n_points, 2), containing all evenly spaced
        points inside the grid boundaries.
    """
    xs = np.linspace(*xlim, num=n_lines)
    ys = np.linspace(*ylim, num=n_points)
    x0, y0 = np.meshgrid(xs, ys)
    lines_x0 = np.atleast_3d(x0.transpose())
    lines_y0 = np.atleast_3d(y0.transpose())

    xs = np.linspace(*xlim, num=n_points)
    ys = np.linspace(*ylim, num=n_lines)
    x1, y1 = np.meshgrid(xs, ys)
    lines_x1 = np.atleast_3d(x1)
    lines_y1 = np.atleast_3d(y1)

    vertical_lines = np.concatenate([lines_x0, lines_y0], axis=2)
    horizontal_lines = np.concatenate([lines_x1, lines_y1], axis=2)

    if n_lines != n_points:
        lines = np.concatenate([vertical_lines, horizontal_lines], axis=0)
    else:
        lines = vertical_lines

    return lines

FeatureSpaceData = namedtuple('FeatureSpaceData', ['line', 'bent_line', 'prediction', 'target'])
FeatureSpaceLines = namedtuple('FeatureSpaceLines', ['grid', 'input', 'contour'])

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

class FeatureSpace(Basic):
    """Creates an instance of a FeatureSpace object to make plots
    and animations.
    Parameters
    ----------
    ax: AxesSubplot
        Subplot of a Matplotlib figure.
    scaled_fixed: boolean, optional
        If True, axis scales are fixed to the maximum from beginning.
        Default is True.
    """
    def __init__(self, ax, scale_fixed=True, boundary=True, cmap=None, alpha=1.0):
        super(FeatureSpace, self).__init__(ax)
        self.ax.grid(False)
        self.scale_fixed = scale_fixed
        self.boundary = boundary
        self.contour = None
        self.bent_inputs = None
        self.bent_lines = None
        self.bent_contour_lines = None
        self.grid_lines = None
        self.contour_lines = None
        self.predictions = None
        self.targets = None
        
        if cmap is None:
            cmap = plt.cm.RdBu
        self.cmap = cmap
        self.alpha = alpha

        self.n_inputs = 0

        self.lines = []
        self.points = []

    def load_data(self, feature_space_data):
        """ Loads feature space data as computed in Replay class.
        Parameters
        ----------
        feature_space_data: FeatureSpaceData
            Namedtuple containing information about original grid
            lines, data points and predictions.
        Returns
        -------
        self: FeatureSpace
            Returns the FeatureSpace instance itself.
        """
        self.predictions = feature_space_data.prediction
        self.targets = feature_space_data.target
        self.grid_lines, self.inputs, self.contour_lines = feature_space_data.line
        self.bent_lines, self.bent_inputs, self.bent_contour_lines = feature_space_data.bent_line

        self.n_epochs = self.bent_inputs.shape[0]
        self.n_inputs = self.bent_inputs.shape[-1]

        self.classes = np.unique(self.targets)
        self.bent_inputs = [self.bent_inputs[:, self.targets == target, :] for target in self.classes]

        self._prepare_plot()
        return self

    def _prepare_plot(self):
        if self.scale_fixed:
            xlim = [self.bent_contour_lines[:, :, :, 0].min() - .05, self.bent_contour_lines[:, :, :, 0].max() + .05]
            ylim = [self.bent_contour_lines[:, :, :, 1].min() - .05, self.bent_contour_lines[:, :, :, 1].max() + .05]
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.ax.set_xlabel(r"$x_0$", fontsize=12)
        self.ax.set_ylabel(r"$x_1$", fontsize=12, rotation=0)

        self.lines = []
        self.points = []
        for c in range(self.grid_lines.shape[0]):
            line, = self.ax.plot([], [], linewidth=0.5, color='k')
            self.lines.append(line)
        for c in range(len(self.classes)):
            point = self.ax.scatter([], [])
            self.points.append(point)

        contour_x = self.bent_contour_lines[0, :, :, 0]
        contour_y = self.bent_contour_lines[0, :, :, 1]
        
        if self.boundary:
            self.contour = self.ax.contourf(contour_x, contour_y, np.zeros(shape=(len(contour_x), len(contour_y))),
                                  cmap=plt.cm.brg, alpha=self.alpha, levels=np.linspace(0, 1, 8))

    @staticmethod
    def _update(i, fs, epoch_start=0, colors=None, **kwargs):
        epoch = i + epoch_start
        fs.ax.set_title('Epoch: {}'.format(epoch))
        if not fs.scale_fixed:
            xlim = [fs.bent_contour_lines[epoch, :, :, 0].min() - .05, fs.bent_contour_lines[epoch, :, :, 0].max() + .05]
            ylim = [fs.bent_contour_lines[epoch, :, :, 1].min() - .05, fs.bent_contour_lines[epoch, :, :, 1].max() + .05]
            fs.ax.set_xlim(xlim)
            fs.ax.set_ylim(ylim)

        if len(fs.lines):
            line_coords = fs.bent_lines[epoch].transpose()

        for c, line in enumerate(fs.lines):
            line.set_data(*line_coords[:, :, c])

        if colors is None:
            colors = ['r', 'b']
            
        if 's' not in kwargs.keys():
            kwargs.update({'s': 10})
            
        if 'marker' not in kwargs.keys():
            kwargs.update({'marker': 'o'})
            
        input_coords = [coord[epoch].transpose() for coord in fs.bent_inputs]
        for c in range(len(fs.points)):
            fs.points[c].remove()
            fs.points[c] = fs.ax.scatter(*input_coords[c], color=colors[int(fs.classes[c])], **kwargs)

        if fs.boundary:
            for c in fs.contour.collections:
                c.remove()  # removes only the contours, leaves the rest intact

            fs.contour = fs.ax.contourf(fs.bent_contour_lines[epoch, :, :, 0],
                                        fs.bent_contour_lines[epoch, :, :, 1],
                                        fs.predictions[epoch].squeeze(),
                                        cmap=fs.cmap, alpha=fs.alpha, levels=np.linspace(0, 1, 8))

        fs.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        fs.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        fs.ax.locator_params(tight=True, nbins=7)
        
        #for tick in fs.ax.xaxis.get_major_ticks():
        #    tick.label.set_fontsize(10)
        #for tick in fs.ax.yaxis.get_major_ticks():
        #    tick.label.set_fontsize(10)
        fs.ax.yaxis.set_label_coords(-0.15,0.5)

        return fs.lines
    

def build_feature_space(model, states, X, y, layer_name=None, contour_points=1000, xlim=(-1, 1), ylim=(-1, 1),
                        display_grid=True, epoch_start=0, epoch_end=-1):
    """Builds a FeatureSpace object to be used for plotting and
    animating.
    The underlying data, that is, grid lines, inputs and contour
    lines, before and after the transformations, as well as the
    corresponding predictions for the contour lines, can be
    later accessed as the second element of the `feature_space`
    property.
    Only layers with 2 hidden units are supported!
    Parameters
    ----------
    ax: AxesSubplot
        Subplot of a Matplotlib figure.
    layer_name: String
        Layer to be used for building the space.
    contour_points: int, optional
        Number of points in each axis of the contour.
        Default is 1,000.
    xlim: tuple of ints, optional
        Boundaries for the X axis of the grid.
    ylim: tuple of ints, optional
        Boundaries for the Y axis of the grid.
    scaled_fixed: boolean, optional
        If True, axis scales are fixed to the maximum from beginning.
        Default is True.
    display_grid: boolean, optional
        If True, display grid lines (for 2-dimensional inputs).
        Default is True.
    epoch_start: int, optional
        First epoch to consider.
    epoch_end: int, optional
        Last epoch to consider.
    Returns
    -------
    feature_space_plot: FeatureSpace
        An instance of a FeatureSpace object to make plots and
        animations.
    """
    layers = list(model.named_modules())
    last_layer_name, last_layer_class = layers[-1]
    is_logit = not isinstance(last_layer_class, nn.Sigmoid)
    if is_logit:
        activation_idx = -2
        func = lambda x: 1 / (1 + np.exp(-x))
    else:
        activation_idx = -3
        func = lambda x: x
    
    names = np.array(list(map(itemgetter(0), layers)))
    matches = names == layer_name

    if np.any(matches):
        activation_idx = np.argmax(matches)
    else:
        raise AttributeError("No layer named {}".format(layer_name))
    if layer_name is None:
        layer_name = layers[activation_idx][0]
    
    try:
        final_dims = layers[activation_idx][1].out_features
    except:
        try:
            final_dims = layers[activation_idx + 1][1].in_features
        except:
            final_dims = layers[activation_idx - 1][1].out_features            
    
    assert final_dims == 2, 'Only layers with 2-dimensional outputs are supported!'

    y_ind = np.atleast_1d(y.squeeze().argsort())
    X = np.atleast_2d(X.squeeze())[y_ind].reshape(X.shape)
    y = np.atleast_1d(y.squeeze())[y_ind]

    if epoch_end == -1:
        epoch_end = len(states)-1
    epoch_end = min(epoch_end, len(states)-1)
    
    #input_dims = self.model.input_shape[-1]
    input_dims = X.shape[-1]
    n_classes = len(np.unique(y))

    # Builds a 2D grid and the corresponding contour coordinates
    grid_lines = np.array([])
    contour_lines = np.array([])
    if input_dims == 2 and display_grid:
        grid_lines = build_2d_grid(xlim, ylim)
        contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points)

    # Initializes "bent" variables, that is, the results of the transformations
    bent_lines = []
    bent_inputs = []
    bent_contour_lines = []
    bent_preds = []
    
    # For each epoch, uses the corresponding weights
    for epoch in range(epoch_start, epoch_end + 1):
        X_values = get_values_for_epoch(model, states, epoch, X)
        bent_inputs.append(X_values[layer_name])
        # Transforms the inputs
        #inputs = [TEST_MODE, X] + weights
        #bent_inputs.append(get_activations(inputs=inputs)[0])

        if input_dims == 2 and display_grid:
            # Transforms the grid lines
            grid_values = get_values_for_epoch(model, states, epoch, grid_lines.reshape(-1, 2))
            #inputs = [TEST_MODE, grid_lines.reshape(-1, 2)] + weights
            output_shape = (grid_lines.shape[:2]) + (-1,)
            #bent_lines.append(get_activations(inputs=inputs)[0].reshape(output_shape))
            bent_lines.append(grid_values[layer_name].reshape(output_shape))

            contour_values = get_values_for_epoch(model, states, epoch, contour_lines.reshape(-1, 2))
            #inputs = [TEST_MODE, contour_lines.reshape(-1, 2)] + weights
            output_shape = (contour_lines.shape[:2]) + (-1,)
            #bent_contour_lines.append(get_activations(inputs=inputs)[0].reshape(output_shape))
            bent_contour_lines.append(contour_values[layer_name].reshape(output_shape))
            # Makes predictions for each point in the contour surface
            #bent_preds.append((get_predictions(inputs=inputs)[0].reshape(output_shape) > .5).astype(int))
            bent_preds.append((func(contour_values[last_layer_name]).reshape(output_shape) > .5).astype(int))
            

    bent_inputs = np.array(bent_inputs)

    # Makes lists into ndarrays and wrap them as namedtuples
    bent_lines = np.array(bent_lines)
    bent_contour_lines = np.array(bent_contour_lines)
    bent_preds = np.array(bent_preds)

    line_data = FeatureSpaceLines(grid=grid_lines, input=X, contour=contour_lines)
    bent_line_data = FeatureSpaceLines(grid=bent_lines, input=bent_inputs, contour=bent_contour_lines)
    _feature_space_data = FeatureSpaceData(line=line_data, bent_line=bent_line_data,
                                                prediction=bent_preds, target=y)

    return _feature_space_data

def build_decision_boundary(model, states, X, y, layer_name=None, contour_points=1000, xlim=(-1, 1), ylim=(-1, 1), display_grid=True,
                            epoch_start=0, epoch_end=-1):
    """Builds a FeatureSpace object to be used for plotting and
    animating the raw inputs and the decision boundary.
    The underlying data, that is, grid lines, inputs and contour
    lines, as well as the corresponding predictions for the
    contour lines, can be later accessed as the second element of
    the  `decision_boundary` property.
    Only inputs with 2 dimensions are supported!
    Parameters
    ----------
    ax: AxesSubplot
        Subplot of a Matplotlib figure.
    contour_points: int, optional
        Number of points in each axis of the contour.
        Default is 1,000.
    xlim: tuple of ints, optional
        Boundaries for the X axis of the grid.
    ylim: tuple of ints, optional
        Boundaries for the Y axis of the grid.
    display_grid: boolean, optional
        If True, display grid lines (for 2-dimensional inputs).
        Default is True.
    epoch_start: int, optional
        First epoch to consider.
    epoch_end: int, optional
        Last epoch to consider.
    Returns
    -------
    decision_boundary_plot: FeatureSpace
        An instance of a FeatureSpace object to make plots and
        animations.
    """
    layers = list(model.named_modules())
    last_layer_name, last_layer_class = layers[-1]
    is_logit = not isinstance(last_layer_class, nn.Sigmoid)
    if is_logit:
        activation_idx = -2
        func = lambda x: 1 / (1 + np.exp(-x))
    else:
        activation_idx = -3
        func = lambda x: x
        
    if layer_name is None:
        layer_name = layers[activation_idx][0]
    else:
        matches = np.array(list(map(itemgetter(0), layers))) == layer_name
        if np.any(matches):
            activation_idx = np.argmax(matches)
        else:
            raise AttributeError("No layer named {}".format(layer_name))

    try:
        final_dims = layers[activation_idx][1].out_features
    except AttributeError:
        final_dims = layers[activation_idx + 1][1].in_features
    assert final_dims == 2, 'Only layers with 2-dimensional outputs are supported!'

    y_ind = y.squeeze().argsort()
    X = X.squeeze()[y_ind].reshape(X.shape)
    y = y.squeeze()[y_ind]

    if epoch_end == -1:
        epoch_end = len(states)-1
    epoch_end = min(epoch_end, len(states)-1)
    
    #input_dims = self.model.input_shape[-1]
    input_dims = X.shape[-1]
    n_classes = len(np.unique(y))
    
    # Builds a 2D grid and the corresponding contour coordinates
    grid_lines = np.array([])
    if display_grid:
        grid_lines = build_2d_grid(xlim, ylim)

    contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points)

    bent_lines = []
    bent_inputs = []
    bent_contour_lines = []
    bent_preds = []
    # For each epoch, uses the corresponding weights
    for epoch in range(epoch_start, epoch_end + 1):
        bent_lines.append(grid_lines)
        bent_inputs.append(X)
        bent_contour_lines.append(contour_lines)

        contour_values = get_values_for_epoch(model, states, epoch, contour_lines.reshape(-1, 2))
        output_shape = (contour_lines.shape[:2]) + (-1,)
        # Makes predictions for each point in the contour surface
        bent_preds.append((func(contour_values[last_layer_name]).reshape(output_shape) > .5).astype(int))

    # Makes lists into ndarrays and wrap them as namedtuples
    bent_inputs = np.array(bent_inputs)
    bent_lines = np.array(bent_lines)
    bent_contour_lines = np.array(bent_contour_lines)
    bent_preds = np.array(bent_preds)

    line_data = FeatureSpaceLines(grid=grid_lines, input=X, contour=contour_lines)
    bent_line_data = FeatureSpaceLines(grid=bent_lines, input=bent_inputs, contour=bent_contour_lines)
    _decision_boundary_data = FeatureSpaceData(line=line_data, bent_line=bent_line_data,
                                                    prediction=bent_preds, target=y)

    return _decision_boundary_data

def get_intermediate_values(model, x):
    hooks = {}
    visualization = {}
    layer_names = {}

    def hook_fn(m, i, o):
        visualization[layer_names[m]] = o.cpu().detach().numpy()
    
    for name, layer in model.named_modules():
        if name != '':
            layer_names[layer] = name
            hooks[name] = layer.register_forward_hook(hook_fn)

    device = list(model.parameters())[0].device.type
    # RNNs
    model(torch.as_tensor(x).float().unsqueeze(0).to(device))
    # model(torch.as_tensor(x).float().to(device))

    for hook in hooks.values():
        hook.remove()
        
    return visualization

def get_values_for_epoch(model, states, epoch, x):
    with torch.no_grad():
        model.load_state_dict(states[epoch])

    return get_intermediate_values(model, x)
