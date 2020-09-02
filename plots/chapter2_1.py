import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def figure1(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])
    ax.set_title('Generated Data - Full Dataset')
    fig.tight_layout()
    return fig