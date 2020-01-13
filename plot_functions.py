# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_timesteps_and_rewards(avg_history):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.subplots_adjust(wspace=0.5)
    axes[0].plot(avg_history['episodes'], avg_history['timesteps'])
    axes[0].set_title('Timesteps in episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Timesteps')
    axes[1].plot(avg_history['episodes'], avg_history['reward'])
    axes[1].set_title('Reward')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward')
    plt.show()


def figure(X=None, Y=None, Z=None, title=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    if X is None:
        X = np.arange(-5, 5, 0.25)
    if Y is None:
        Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    if Z is None:
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
    if title is None:
        title = 'sin surface'
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()
