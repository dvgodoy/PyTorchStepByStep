# %% [markdown]
# # Deep Learning with PyTorch Step-by-Step: A Beginner's Guide

# %% [markdown]
# # Chapter 0

# %%
"""
try:
    import google.colab
    import requests
    url = 'https://raw.githubusercontent.com/dvgodoy/PyTorchStepByStep/master/config.py'
    r = requests.get(url, allow_redirects=True)
    open('config.py', 'wb').write(r.content)    
except ModuleNotFoundError:
    pass
"""
from config import *


# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from plots import chapter0


# %% [markdown]
# # Visualizing Gradient Descent

# %% [markdown]
# ## Model

# %% [markdown]
# $$
# \Large y = b + w x + \epsilon
# $$

# %% [markdown]
# ## Data Generation

# %% [markdown]
# ### Synthetic Data Generation

# %%
true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

# %% [markdown]
# ### Train-Validation-Test Split

# %%
# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# %%
chapter0.figure1(x_train, y_train, x_val, y_val)

# %% [markdown]
# ## Step 0: Random Initialization

# %%
# Step 0 - Initializes parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b, w)

# %% [markdown]
# ## Step 1: Compute Model's Predictions

# %%
# Step 1 - Computes our model's predicted output - forward pass
yhat = b + w * x_train

# %%
chapter0.figure2(x_train, y_train, b, w)

# %% [markdown]
# ## Step 2: Compute the Loss

# %% [markdown]
# $$
# \Large \text{error}_i = \hat{y_i} - y_i
# $$

# %%
chapter0.Figure3(x_train, y_train, b, w)

# %% [markdown]
# $$
# \Large
# \begin{aligned}
# \text{MSE} &= \frac{1}{n} \sum_{i=1}^n{\text{error}_i}^2
# \\
# &= \frac{1}{n} \sum_{i=1}^n{(\hat{y_i} - y_i)}^2
# \\
# &= \frac{1}{n} \sum_{i=1}^n{(b + w x_i - y_i)}^2
# \end{aligned}
# $$

# %%
# Step 2 - Computing the loss
# We are using ALL data points, so this is BATCH gradient
# descent. How wrong is our model? That's the error!
error = (yhat - y_train)

# It is a regression, so it computes mean squared error (MSE)
loss = (error ** 2).mean()
print(loss)

# %% [markdown]
# ### Loss Surface

# %%
# Reminder:
# true_b = 1
# true_w = 2

# we have to split the ranges in 100 evenly spaced intervals each
b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)
# meshgrid is a handy function that generates a grid of b and w
# values for all combinations
bs, ws = np.meshgrid(b_range, w_range)
bs.shape, ws.shape

# %%
bs

# %%
sample_x = x_train[0]
sample_yhat = bs + ws * sample_x
sample_yhat.shape

# %%
all_predictions = np.apply_along_axis(
    func1d=lambda x: bs + ws * x, 
    axis=1, 
    arr=x_train
)
all_predictions.shape

# %%
all_labels = y_train.reshape(-1, 1, 1)
all_labels.shape

# %%
all_errors = (all_predictions - all_labels)
all_errors.shape

# %%
all_losses = (all_errors ** 2).mean(axis=0)
all_losses.shape

# %%
chapter0.Figure4(x_train, y_train, b, w, bs, ws, all_losses)

# %% [markdown]
# ### Cross Sections

# %%
chapter0.Figure5(x_train, y_train, b, w, bs, ws, all_losses)

# %%
chapter0.Figure6(x_train, y_train, b, w, bs, ws, all_losses)

# %% [markdown]
# ## Step 3: Compute the Gradients

# %% [markdown]
# $$
# \Large
# \begin{aligned}
# \frac{\partial{\text{MSE}}}{\partial{b}} = \frac{\partial{\text{MSE}}}{\partial{\hat{y_i}}} \frac{\partial{\hat{y_i}}}{\partial{b}} &= \frac{1}{n} \sum_{i=1}^n{2(b + w x_i - y_i)} 
# \\
# &= 2 \frac{1}{n} \sum_{i=1}^n{(\hat{y_i} - y_i)}
# \\
# \frac{\partial{\text{MSE}}}{\partial{w}} = \frac{\partial{\text{MSE}}}{\partial{\hat{y_i}}} \frac{\partial{\hat{y_i}}}{\partial{w}} &= \frac{1}{n} \sum_{i=1}^n{2(b + w x_i - y_i) x_i} 
# \\
# &= 2 \frac{1}{n} \sum_{i=1}^n{x_i (\hat{y_i} - y_i)}
# \end{aligned}
# $$

# %%
# Step 3 - Computes gradients for both "b" and "w" parameters
b_grad = 2 * error.mean()
w_grad = 2 * (x_train * error).mean()
print(b_grad, w_grad)

# %% [markdown]
# ### Visualizing the Gradients

# %%
chapter0.Figure7(b, w, bs, ws, all_losses)

# %%
chapter0.Figure8(b, w, bs, ws, all_losses)

# %% [markdown]
# ### Backpropagation

# %% [markdown]
# ## Step 4: Update the Parameters

# %% [markdown]
# $$
# \Large
# \begin{aligned}
# b &= b - \eta \frac{\partial{\text{MSE}}}{\partial{b}}
# \\
# w &= w - \eta \frac{\partial{\text{MSE}}}{\partial{w}}
# \end{aligned}
# $$

# %%
# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.1
print(b, w)

# Step 4 - Updates parameters using gradients and the 
# learning rate
b = b - lr * b_grad
w = w - lr * w_grad

print(b, w)

# %%
chapter0.Figure9(x_train, y_train, b, w)

# %% [markdown]
# ### Learning Rate

# %%
manual_grad_b = -2.90
manual_grad_w = -1.79

np.random.seed(42)
b_initial = np.random.randn(1)
w_initial = np.random.randn(1)

# %% [markdown]
# #### Low Learning Rate

# %%
# Learning rate - greek letter "eta" that looks like an "n"
lr = .2

figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr)

# %% [markdown]
# #### High Learning Rate

# %%
# Learning rate - greek letter "eta" that looks like an "n"
lr = .8

figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr)

# %% [markdown]
# #### Very High Learning Rate

# %%
# Learning rate - greek letter "eta" that looks like an "n"
lr = 1.1

figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr)

# %% [markdown]
# #### "Bad" Feature

# %%
true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)

# We divide w by 10
bad_w = true_w / 10
# And multiply x by 10
bad_x = np.random.rand(N, 1) * 10

# So, the net effect on y is zero - it is still
# the same as before
y = true_b + bad_w * bad_x + (.1 * np.random.randn(N, 1))

# %%
# Generates train and validation sets
# It uses the same train_idx and val_idx as before,
# but it applies to bad_x
bad_x_train, y_train = bad_x[train_idx], y[train_idx]
bad_x_val, y_val = bad_x[val_idx], y[val_idx]

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(x_train, y_train)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_ylim([0, 3.1])
ax[0].set_title('Train - Original')
ax[1].scatter(bad_x_train, y_train, c='k')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_ylim([0, 3.1])
ax[1].set_title('Train - "Bad"')
fig.tight_layout()

# %%
# The ranges CHANGED because we are centering at the new minimum, using "bad" data
bad_b_range = np.linspace(-2, 4, 101)
bad_w_range = np.linspace(-2.8, 3.2, 101)
bad_bs, bad_ws = np.meshgrid(bad_b_range, bad_w_range)

# %%
figure14(x_train, y_train, b_initial, w_initial, bad_bs, bad_ws, bad_x_train)

# %%
figure15(x_train, y_train, b_initial, w_initial, bad_bs, bad_ws, bad_x_train)

# %% [markdown]
# #### Scaling / Standardizing / Normalizing

# %% [markdown]
# $$
# \Large
# \overline{X} = \frac{1}{N}\sum_{i=1}^N{x_i}
# \\
# \Large
# \sigma(X) = \sqrt{\frac{1}{N}\sum_{i=1}^N{(x_i - \overline{X})^2}}
# \\
# \Large
# \text{scaled } x_i=\frac{x_i-\overline{X}}{\sigma(X)}
# $$

# %%
scaler = StandardScaler(with_mean=True, with_std=True)
# We use the TRAIN set ONLY to fit the scaler
scaler.fit(x_train)

# Now we can use the already fit scaler to TRANSFORM
# both TRAIN and VALIDATION sets
scaled_x_train = scaler.transform(x_train)
scaled_x_val = scaler.transform(x_val)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax[0].scatter(x_train, y_train, c='b')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_ylim([0, 3.1])
ax[0].set_title('Train - Original')
ax[1].scatter(bad_x_train, y_train, c='k')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_ylim([0, 3.1])
ax[1].set_title('Train - "Bad"')
ax[1].label_outer()
ax[2].scatter(scaled_x_train, y_train, c='g')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].set_ylim([0, 3.1])
ax[2].set_title('Train - Scaled')
ax[2].label_outer()

fig.tight_layout()

# %%
# The ranges CHANGED AGAIN because we are centering at the new minimum, using "scaled" data
scaled_b_range = np.linspace(-1, 5, 101)
scaled_w_range = np.linspace(-2.4, 3.6, 101)
scaled_bs, scaled_ws = np.meshgrid(scaled_b_range, scaled_w_range)

# %%
figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train)

# %% [markdown]
# ## Step 5: Rinse and Repeat!

# %%
figure18(x_train, y_train)

# %% [markdown]
# ### The Path of Gradient Descent

# %% [markdown]
# Even though the plots are important to illustrate the paths, the corresponding code is beyond the scope of this chapter.

# %% [markdown]
# ![](images/paths.png)

# %%



