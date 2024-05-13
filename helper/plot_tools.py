def pretty(ax):
    # remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # set limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    return


def format_iteration(i: int):
    """ Format the iteration number """
    if i < 9:
        return f"000{i}"
    elif i < 99:
        return f"00{i}"
    elif i < 999:
        return f"0{i}"
    else:
        return f"{i}"


def draw_particles(particles: np.ndarray, time: float):
    """
    Draw particles with color based on the potential energy
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    # draw particles, color based on the timestep
    colors = time * np.ones(len(particles))
    # convert time from [0,1] to viridis colormap
    colors = plt.cm.viridis(colors)
    # thin edge
    ax.scatter(particles[:, 0], particles[:, 1], c=colors, edgecolors='black', s=25, linewidth=0.5, alpha=0.5)
    pretty(ax)
    return fig, ax
