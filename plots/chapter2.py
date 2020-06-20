import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def plot_losses(losses, val_losses):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(losses, label='Training Loss', c='b')
    plt.plot(val_losses, label='Validation Loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    return fig


def plot_resumed_losses(saved_epoch, saved_losses, saved_val_losses, n_epochs, losses, val_losses):
    range_before = range(0, saved_epoch)
    range_after = range(saved_epoch, saved_epoch + n_epochs)

    fig = plt.figure(figsize=(10, 4))
    # Checkpointed loss
    plt.plot(range_before, saved_losses, 
             label='Checkpointed Training Loss', c='b', linestyle='--')
    plt.plot(range_before, saved_val_losses, 
             label='Checkpointed Validation Loss', c='r', linestyle='--')
    # Losses after resuming
    plt.plot(range_after, losses, label='Training Loss', c='b')
    plt.plot(range_after, val_losses, label='Validation Loss', c='r')
    # Divider
    plt.plot([saved_epoch, saved_epoch],
             [np.min(saved_losses + losses), np.max(saved_losses + losses)], 
             c='k', linewidth=1, linestyle='--', label='Checkpoint')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    return fig