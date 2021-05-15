import math
import numpy as np
import matplotlib
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Arc

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
        
#     if end_ind > 1:
#         end_ind = 0
#     if end_ind < 0:
#         end_ind = 1

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

def sequence_pred(sbs_obj, X, directions=None, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axs = axs.flatten()

    for e, ax in enumerate(axs):
        first_corners = X[e, :2, :]
        sbs_obj.model.eval()
        next_corners = sbs_obj.model(X[e:e+1, :2].to(sbs_obj.device)).squeeze().detach().cpu().numpy()
        pred_corners = np.concatenate([first_corners, next_corners], axis=0)

        for j, corners in enumerate([X[e], pred_corners]):
            for i in range(4):
                coords = corners[i]
                color = 'k'
                ax.scatter(*coords, c=color, s=400)
                if i == 3:
                    start = -1
                else:
                    start = i
                if (not j) or (j and i):
                    ax.plot(*corners[[start, start+1]].T, c='k', lw=2, alpha=.5, linestyle='--' if j else '-')
                ax.text(*(coords - np.array([.04, 0.04])), str(i+1), c='w', fontsize=12)
                if directions is not None:
                    ax.set_title(f'{"Counter-" if not directions[e] else ""}Clockwise')

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$", rotation=0)
        ax.set_xlim([-1.45, 1.45])
        ax.set_ylim([-1.45, 1.45])        

    fig.tight_layout()
    return fig

# https://stackoverflow.com/questions/25227100/best-way-to-plot-an-angle-between-two-lines-in-matplotlib
# https://gist.github.com/battlecook/0c0bdb7097ec7c8fa160e342b1bf51ef
def get_angle_plot(line1, line2, offset=1, color=None, origin=(0, 0), len_x_axis = 1, len_y_axis = 1):

    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][1]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1))) # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][1] - l2xy[0][1]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    angle_plot = Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color=color, label = str(angle)+u"\u00b0")
    angle = angle_plot.get_label()[:-1] # Excluding the degree symbol
    angle_text = "%0.2f"%float(angle)+u"\u00b0" # Display angle upto 2 decimal places
    
    return angle_plot, angle_text

def make_line(ax, point):
    point = np.vstack([[0., 0.], np.array(point.squeeze().tolist())])
    line = ax.plot(*point.T, lw=0)[0]
    return line

def project_cosine(q, k):
    norm_q, norm_k = np.linalg.norm(q), np.linalg.norm(k)
    unit_q = q / norm_q
    unit_k = k / norm_k
    cos_theta = np.dot(unit_q, unit_k)
    unit_proj = cos_theta * unit_k
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    ax = axs[0]
    line_q = make_line(ax, q)
    line_k = make_line(ax, k)
    add_arrow(line_q, lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
    add_arrow(line_k, lw=2, color='k', text=f'||K||={norm_k:.2f}', size=12, text_offset=(-.03, .02))
    circle1 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle1)

    angle_plot, angle_text = get_angle_plot(line_q, line_k, offset=.3, color='k')
    ax.add_patch(angle_plot) # To display the angle arc
    ax.text(.2, .2, angle_text)

    ax = axs[1]
    add_arrow(make_line(ax, q / norm_q), lw=2, color='r', text=r'$||\hat{Q}||=1$', size=12, alpha=.9, text_offset=(.02, .02))
    add_arrow(make_line(ax, k / norm_k), lw=2, color='k', text=r'$||\hat{K}||=1$', size=12, alpha=.9, text_offset=(.02, .02))
    circle2 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle2)

    angle_plot, angle_text = get_angle_plot(line_q, line_k, offset=.3, color='k')
    ax.add_patch(angle_plot) # To display the angle arc
    ax.text(.2, .2, angle_text)

    ax = axs[2]
    add_arrow(make_line(ax, q / norm_q), lw=2, color='r', text=r'$\hat{Q}$', size=12, alpha=.9, text_offset=(.02, .02))
    add_arrow(make_line(ax, k / norm_k), lw=2, color='k', text=r'$\hat{K}$', size=12, alpha=.9, text_offset=(.02, .02))
    circle3 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle3)
    add_arrow(make_line(ax, unit_proj), lw=2, color='r', 
              text=r'$||\hat{Q}_\hat{K}||=cos\theta=' + f'{cos_theta:.2f}' + '$', size=12, alpha=.9, text_offset=(-.35, -.16))

    angle_plot, angle_text = get_angle_plot(line_q, line_k, offset=.3, color='k')
    ax.add_patch(angle_plot) # To display the angle arc
    ax.text(.2, .2, angle_text)

    ax.plot(*np.stack([q / norm_q, unit_proj]).T, lw=2, linestyle='--', color='gray')

    titles = [r'$Q\ and\ K$',
              r'$Unit\ Norm: \hat{Q}\ and\ \hat{K}$',
              r'$Projection: \hat{Q}_{\hat{K}}$']
    for i, ax in enumerate(axs):
        ax.set_ylim([0, 1.02])
        ax.set_xlim([0, 1.02])
        ax.set_xticklabels([0., .5, 1.0], fontsize=12)
        ax.set_yticks([0, .5, 1.0])
        ax.set_yticklabels([0., .5, 1.0], fontsize=12)
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$x_1$')
        ax.set_title(titles[i])

    fig.tight_layout()
    return fig

def project_cosine_scaling(q, k):
    norm_q, norm_k = np.linalg.norm(q), np.linalg.norm(k)
    unit_q = q / norm_q
    unit_k = k / norm_k
    cos_theta = np.dot(unit_q, unit_k)
    unit_proj = cos_theta * unit_k

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    ax = axs[0]
    add_arrow(make_line(ax, unit_proj), lw=2, color='r', 
              text=r'$cos\theta=' + f'{cos_theta:.2f}' + '$', size=12, alpha=.9, text_offset=(-.13, .03))

    ax = axs[1]
    add_arrow(make_line(ax, q), lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
    add_arrow(make_line(ax, unit_proj * norm_q), lw=2, color='r', 
              text=r'$cos\theta\ ||Q||=' + f'{cos_theta*norm_q:.2f}' + '$', size=12, alpha=.9, text_offset=(-.18, .03))

    ax = axs[2]
    add_arrow(make_line(ax, q), lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
    add_arrow(make_line(ax, k), lw=2, color='k', text=f'||K||={norm_k:.2f}', size=12, text_offset=(-.03, .02))
    add_arrow(make_line(ax, unit_proj * norm_q * norm_k), lw=2, color='r', 
              text=r'$cos\theta\ ||Q||\ ||K||=' + f'{cos_theta*norm_q*norm_k:.2f}' + '$', 
              size=12, alpha=.9, text_offset=(-.23, -.13))

    titles = [r'$Projection: \hat{Q}_{\hat{K}}$', 
              r'$Scaling\ by\ ||Q||$',
              r'$Scaling\ by\ ||K||$']
    for i, ax in enumerate(axs):
        ax.set_ylim([0, 1.02])
        ax.set_xlim([0, 1.02])
        ax.set_xticks([0, .5, 1.0])
        ax.set_xticklabels([0., .5, 1.0], fontsize=12)
        ax.set_yticks([0, .5, 1.0])
        ax.set_yticklabels([0., .5, 1.0], fontsize=12)
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$x_1$')
        ax.set_title(titles[i])

    fig.tight_layout()
    return fig

def query_and_keys(q, ks, result=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.get_figure()

    norm_q = np.linalg.norm(q)
    line_q = make_line(ax, q)
    
    line_k = []
    norm_k = []
    cos_k = []
    for k in ks:
        line_k.append(make_line(ax, k))
        norm_k.append(np.linalg.norm(k))
        cos_k.append(np.dot(q, k)/(norm_k[-1]*norm_q))

    add_arrow(line_q, lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
    add_arrow(line_k[0], lw=2, color='k', text=r'$||K_0' + f'||={norm_k[0]:.2f}$' + '\n' + r'$cos\theta_0=' + f'{cos_k[0]:.2f}$', size=12, text_offset=(-.33, .1))
    add_arrow(line_k[1], lw=2, color='k', text=r'$||K_1' + f'||={norm_k[1]:.2f}$' + '\n' + r'$cos\theta_1=' + f'{cos_k[1]:.2f}$', size=12, text_offset=(-.63, -.18))
    add_arrow(line_k[2], lw=2, color='k', text=r'$||K_2' + f'||={norm_k[2]:.2f}$' + '\n' + r'$cos\theta_2=' + f'{cos_k[2]:.2f}$', size=12, text_offset=(.05, .58))
    if result is not None:
#         add_arrow(line_q, lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
#         add_arrow(line_k0, lw=2, color='k', text=r'$||K_0' + f'||={norm_k0:.2f}$', size=12, text_offset=(-.2, .1))
#         add_arrow(line_k1, lw=2, color='k', text=r'$||K_1' + f'||={norm_k1:.2f}$', size=12, text_offset=(-.53, -.12))
#         add_arrow(line_k2, lw=2, color='k', text=r'$||K_2' + f'||={norm_k2:.2f}$', size=12, text_offset=(-.08, -.14))        
        add_arrow(make_line(ax, result), lw=2, color='g', text=f'Context Vector', size=12, text_offset=(-.26, .1))
    circle1 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle1)

    ax.set_ylim([-1.02, 1.02])
    ax.set_xlim([-1.02, 1.02])

    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_xticklabels([-1.0, 0, 1.0], fontsize=12)
    ax.set_yticks([-1.0, 0, 1.0])
    ax.set_yticklabels([-1.0, 0, 1.0], fontsize=12)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'$Query\ and\ Keys$')
    fig.tight_layout()
    return fig

def plot_attention(model, inputs, point_labels=None, source_labels=None, target_labels=None, decoder=False, self_attn=False, n_cols=5, alphas_attr='alphas'):
    textcolors=["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    model.eval()
    device = list(model.parameters())[0].device.type
    predicted_seqs = model(inputs.to(device))
    alphas = model
    for attr in alphas_attr.split('.'):
        alphas = getattr(alphas, attr)
    if len(alphas.shape) < 4:
        alphas = alphas.unsqueeze(0)
    alphas = np.array(alphas.tolist())
    n_heads, n_points, target_len, input_len = alphas.shape
    
    if point_labels is None:
        point_labels = [f'Point #{i}' for i in range(n_points)]
        
    if source_labels is None:
        source_labels = [f'Input #{i}' for i in range(input_len)]

    if target_labels is None:
        target_labels = [f'Target #{i}' for i in range(target_len)]
            
    if self_attn:
        if decoder:
            source_labels = source_labels[-1:] + target_labels[:-1]
        else:
            target_labels = source_labels        
        
    if n_heads == 1:
        n_rows = (n_points // n_cols) + int((n_points % n_cols) > 0)
    else:
        n_cols = n_heads
        n_rows = n_points

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))

    for i in range(n_points):
        for head in range(n_heads):
            data = alphas[head][i].squeeze()
            if n_heads > 1:
                if n_points > 1:
                    ax = axs[i, head]
                else:
                    ax = axs[head]
            else:
                ax = axs.flat[i]

            im = ax.imshow(data, vmin=0, vmax=1, cmap=plt.cm.gray)
            ax.grid(False)

            #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
            
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            
            ax.set_xticklabels(source_labels)
            if n_heads == 1:
                ax.set_title(point_labels[i], fontsize=14)
            else:
                if i == 0:
                    ax.set_title(f'Attention Head #{head+1}', fontsize=14)
                if head == 0:
                    ax.set_ylabel(point_labels[i], fontsize=14)

            ax.set_yticklabels([])
            if n_heads == 1:
                if not (i % n_cols):
                    ax.set_yticklabels(target_labels)
            else:
                if head == 0:
                    ax.set_yticklabels(target_labels)            

            ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            threshold = im.norm(data.max())/2.
            for ip in range(data.shape[0]):
                for jp in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[ip, jp]) > threshold)])
                    text = im.axes.text(jp, ip, valfmt(data[ip, jp], None), **kw)

    fig.subplots_adjust(wspace=0.8, hspace=1.0)
    fig.tight_layout()
    return fig

def figure9():
    english = ['the', 'European', 'economic', 'zone']
    french = ['la', 'zone',  'économique', 'européenne']

    source_labels = english
    target_labels = french

    data = np.array([[.8, 0, 0, .2],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, .8, 0, .2]])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap=plt.cm.gray)
    ax.grid(False)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(source_labels, rotation=90)

    ax.set_yticklabels([])
    ax.set_yticklabels(target_labels)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    textcolors=["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    threshold = im.norm(data.max())/2.
    for ip in range(data.shape[0]):
        for jp in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[ip, jp]) > threshold)])
            text = im.axes.text(jp, ip, valfmt(data[ip, jp], None), **kw)

    fig.tight_layout()
    return fig

def plot_dial(xs, ys, seq, dim, scale, ax, has_coords=False):
    point = np.array([xs[1, 0], ys[1, 0]])
    point = np.vstack([point*.9, point*1.1])
    line_tick = ax.plot(*point.T, lw=2, c='k')[0]
    line_zero = ax.plot([0.9, 1.1], [0, 0], lw=2, c='k')[0]

    q = np.array([xs[seq, dim], ys[seq, dim]])

    line_q = make_line(ax, q*.95)  
    add_arrow(line_q, lw=2, color='r', text='', size=12)
    circle1 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle1)

    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-1.1, 1.1])
    ax.grid(False)
    ax.text(xs[1, 0]+np.sign(xs[1, 0])*.1-.2, ys[1, 0]+(.15 if ys[1,0]>.001 else -.05), scale)
    ax.text(1.2, -.05, '0')
    if has_coords:
        ax.set_xlabel(f'({ys[seq, dim]:.2f}, {xs[seq, dim]:.2f})')
    ax.set_xticks([])
    ax.set_yticks([])
    
def plot_text(x, y, text, ax, fontsize=24):
    ax.text(x, y, text, fontsize=fontsize)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, 1])
    
def gen_coords(d_model, max_len, exponent=10000):
    position = torch.arange(0, max_len).float().unsqueeze(1)
    angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(exponent) / d_model))
    return np.cos(position * angular_speed), np.sin(position * angular_speed)

def encoding_degrees(dims, seqs, tot):
    fig, axs = plt.subplots(dims+1, tot+1, figsize=(2*(tot+1), 2*(1+dims)))
    axs = np.atleast_2d(axs)
    for dim in range(dims):
        xs, ys = np.cos(np.linspace(0, tot/seqs[dim], tot+1)*2*np.pi).reshape(-1, 1), np.sin(np.linspace(0, tot/seqs[dim], tot+1)*2*np.pi).reshape(-1, 1)
        for seq in range(tot):
            plot_text(-.1, -.5, seq, axs[0, seq+1])
            plot_text(-.5, -.5, 'Position', axs[0, 0])
            if seq == 0:
                plot_text(-.5, -.2, f'Base {seqs[dim]}', axs[dim+1, 0])
                plot_text(-.5, -1.3, f'(sine, cosine)', axs[dim+1, 0], fontsize=16)
            plot_dial(xs, ys, seq=seq, dim=0, scale=f'1/{seqs[dim]}', ax=axs[dim+1, seq+1], has_coords=True)
        seqs *= 2

    fig.tight_layout()
    return fig

def exponential_dials(d_model, max_len):
    xs, ys = gen_coords(d_model, max_len)
    dims = int(d_model/2)
    seqs = max_len
    fig, axs = plt.subplots(dims+1, seqs+1, figsize=(2*(1+seqs), 2*(1+dims)))
    axs = np.atleast_2d(axs)
    for seq in range(seqs):
        plot_text(-.1, -.5, seq, axs[0, seq+1])
        plot_text(-.5, -.5, 'Position', axs[0, 0])
        for dim in range(dims):
            scale = 10**dim
            if seq == 0:
                plot_text(-.5, -.2, f'Dims #{2*dim},#{2*dim+1}', axs[dim+1, 0], fontsize=16)
                plot_text(-.5, -1.35, f'(sine, cosine)', axs[dim+1, 0], fontsize=16)
            plot_dial(xs, ys, seq=seq, dim=dim, scale=scale, ax=axs[dim+1, seq+1], has_coords=True)

    fig.tight_layout()
    return fig

def plot_mesh(values, ax, showvals=False, colorbar=False, ylabel=None):
    max_len, d_model = values.squeeze().shape
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    else:
        fig = ax.get_figure()
    im = ax.pcolormesh(values.T, cmap='viridis')
    if ylabel is None:
        ylabel = 'Dimensions'
    ax.set_ylabel(ylabel)
    ax.set_ylim((d_model-.5, 0))
    ax.set_yticks(np.arange(d_model+1, -1, -1)-.5)
    ax.set_yticklabels([''] + list(range(d_model-1, -1, -1)) + [''])
    ax.set_xlabel('Positions')
    ax.set_xlim((0, max_len))
    ax.set_xticks(np.arange(0, max_len+1)-.5)
    ax.set_xticklabels([''] + list(range(0, max_len)))
    if colorbar:
        plt.colorbar(im)
        
    if showvals:
        textcolors=["white", "black"]
        kw = dict(horizontalalignment="center", verticalalignment="center")
        valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
        threshold = im.norm(values.max())/2.
        for ip in range(values.shape[0]):
            for jp in range(values.shape[1]):
                kw.update(color=textcolors[int(im.norm(values[ip, jp]) > threshold)])
                text = im.axes.text(jp+.5, ip+.5, valfmt(values[ip, jp], None), **kw)        
    
    fig.tight_layout()
    return fig

def encoding_heatmap(d_model, max_len):
    position = torch.arange(0, max_len).float().unsqueeze(1)
    angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

    encoding = torch.zeros(max_len, d_model)
    encoding[:, 0::2] = torch.sin(angular_speed * position)
    encoding[:, 1::2] = torch.cos(angular_speed * position)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    sines = torch.sin(angular_speed * position)
    cosines = torch.cos(angular_speed * position)

    axs = axs.flatten()
    axs[0].plot(sines)
    axs[0].set_title('Sines - Even Dimensions')
    axs[0].legend(["dim %d" % p for p in range(0, 8, 2)], loc='lower left')

    axs[1].plot(cosines)
    axs[1].set_title('Cosines - Odd Dimensions')
    axs[1].legend(["dim %d" % p for p in range(1, 9, 2)], loc='lower left')

    values = np.zeros((max_len, d_model))
    values.fill(np.nan)
    values[:, 0::2] = sines

    fig = plot_mesh(values, axs[2])

    values = np.zeros((max_len, d_model))
    values.fill(np.nan)
    values[:, 1::2] = cosines
    fig = plot_mesh(values, axs[3])
    
    return fig
