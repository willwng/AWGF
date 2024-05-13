from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from tqdm import tqdm
import csv


def potential(x: np.ndarray) -> float:
    """
    Potential energy of a single particle
    :param x: position of the particle
    :return: potential energy of the particle
    """
    return np.square(np.linalg.norm(x))


def d_potential(x: np.ndarray) -> np.ndarray:
    """
    Gradient of the potential energy of a single particle
    :param x: position of the particle
    :return: gradient of the potential energy of the particle
    """
    return 2 * x


def potential_particles(particles: np.ndarray) -> float:
    """
    Lagrangian discretization of the potential energy
    :param particles: list of positions of particles
    :return: potential energy of the system
    """
    n = len(particles)
    return (1 / n) * np.sum(np.vectorize(potential)(particles))


def nabla_potential_particles(particles: np.ndarray) -> np.ndarray:
    """
    Gradient of the potential energy of the system
    :param particles: list of positions of particles
    :return: gradient of the potential energy of the system
    """
    n = len(particles)
    return (1 / n) * np.vectorize(d_potential)(particles)


def get_nn(p_i: np.ndarray, particles: np.ndarray, tree) -> np.ndarray:
    """
    Find the nearest neighbor distance of a particle i in a list of particles (not including p_i)
    :param p_i: position of the particle i
    :param tree: KDTree of the particles
    :return: position of the nearest neighbor
    """
    nns = tree.query(x=p_i, k=2, p=2)
    return particles[nns[1][1]]  # return the second-nearest neighbor position


def entropy_particles(particles: np.ndarray, tree) -> float:
    """
    Lagrangian discretization of the entropy
    :param particles: list of positions of particles
    :return: entropy of the system
    """
    n = len(particles)
    # Create a KDTree for fast nearest neighbor search
    entropy = 0
    for i, p_i in enumerate(particles):
        p_j = get_nn(p_i, particles, tree)
        d_ij = np.linalg.norm(p_i - p_j)
        entropy += np.log(d_ij)
    return (1 / n) * entropy


def nabla_entropy_particles(particles: np.ndarray, tree) -> np.ndarray:
    """
    Gradient of the entropy of the system
    :param particles: list of positions of particles
    :return: gradient of the entropy of the system
    """
    n = len(particles)
    # Create a KDTree for fast nearest neighbor search
    grad_entropy = np.zeros_like(particles)
    for i, p_i in enumerate(particles):
        p_j = get_nn(p_i, particles, tree)
        d_ij = np.linalg.norm(p_i - p_j)
        # derivative of d_ij with respect to particle i
        d_d_ij = (p_i - p_j) / d_ij
        grad_entropy[i] = (1 / d_ij) * d_d_ij
    return (1 / n) * grad_entropy


def sample_distribution(n_dimensions: int, low: float, high: float) -> np.ndarray:
    """
    Sample a random position in n_dimensions from a uniform distribution
    """
    return np.random.uniform(low, high, (n_dimensions,))


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


def get_total_energy_gradient(particles: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Get the total gradient of the system
    """
    tree = scipy.spatial.cKDTree(particles)
    grad_potential = nabla_potential_particles(particles)
    grad_entropy = nabla_entropy_particles(particles, tree)
    # V = int x^2 rho - H(rho)
    grad_total = grad_potential - grad_entropy
    energy_total = potential_particles(particles) - entropy_particles(particles, tree)
    return energy_total, grad_total


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


def main():
    n_particles = 1000
    n_dimensions = 2
    # get positions of particles
    particles = np.zeros((n_particles, n_dimensions))
    # First half of particles from (-5, -5) to (0, 0), second half from (0, 0) to (5, 5)
    for i in range(n_particles // 2):
        particles[i] = sample_distribution(n_dimensions, -5, 0)
    for i in range(n_particles // 2, n_particles):
        particles[i] = sample_distribution(n_dimensions, 0, 5)

    draw_particles(particles, time=0.0)
    plt.savefig("initial_configuration.pdf")
    energies = []
    # Optimize the system
    max_iter = 5000
    progress_bar = tqdm(range(max_iter))
    for i in progress_bar:
        if i % 25 == 0:
            draw_particles(particles, time=float(i) / max_iter)
            plt.savefig(f"out/iteration_{format_iteration(i)}.png", dpi=300)
        energy_total, grad_total = get_total_energy_gradient(particles)
        particles -= 0.5 * grad_total
        energies.append(energy_total)
        progress_bar.set_description(f"Energy: {energy_total:.4f}")

    # Save the energies
    with open("out/energies.csv", "w") as f:
        writer = csv.writer(f)
        for i, e in enumerate(energies):
            writer.writerow([i, e])

    draw_particles(particles, time=1.0)
    plt.savefig("final_configuration.pdf")


if __name__ == '__main__':
    main()
