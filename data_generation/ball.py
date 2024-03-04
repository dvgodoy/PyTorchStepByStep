import numpy as np

def load_data(n_dims=10, n_points=1000, classif_radius_fraction=0.5, only_sphere=False, shuffle=True, seed=13):
    """
    Parameters
    ----------
    n_dims: int, optional
        Number of dimensions of the n-ball. Default is 10.
    n_points: int, optional
        Number of points in each parabola. Default is 1,000.
    classif_radius_fraction: float, optional
        Points farther away from the center than
        `classification_radius_fraction * ball radius` are
        considered to be positive cases. The remaining
        points are the negative cases.
    only_sphere: boolean
        If True, generates a n-sphere, that is, a hollow n-ball.
        Default is False.
    shuffle: boolean, optional
        If True, the points are shuffled. Default is True.
    seed: int, optional
        Random seed. Default is 13.
    Returns
    -------
    X, y: tuple of ndarray
        X is an array of shape (n_points, n_dims) containing the
        points in the n-ball.
        y is an array of shape (n_points, 1) containing the
        classes of the samples.
    """
    np.random.seed(seed)
    radius = np.sqrt(n_dims)
    points = np.random.normal(size=(n_points, n_dims))
    sphere = radius * points / np.linalg.norm(points, axis=1).reshape(-1, 1)
    if only_sphere:
        X = sphere
    else:
        X = sphere * np.random.uniform(size=(n_points, 1))**(1 / n_dims)

    adjustment = 1 / np.std(X)
    radius *= adjustment
    X *= adjustment

    y = (np.abs(np.sum(X, axis=1)) > (radius * classif_radius_fraction)).astype(int)

    # But we must not feed the network with neatly organized inputs...
    # so let's randomize them
    if shuffle:
        np.random.seed(seed)
        shuffled = np.random.permutation(range(X.shape[0]))
        X = X[shuffled]
        y = y[shuffled].reshape(-1, 1)

    return (X, y)
