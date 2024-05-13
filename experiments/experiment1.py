import csv
from typing import Tuple

import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

from helper.distribution_helper import sample_distribution
from helper.plot_tools import *

from energies.entropy import *
from energies.quadratic import *


def total_energy_gradient(particles: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Experiment 1: Combination of quadratic linear term and entropy
    """
    tree = scipy.spatial.cKDTree(particles)
    grad_potential = nabla_quad_potential_particles(particles)
    grad_entropy = nabla_entropy_particles(particles, tree)
    grad_total = grad_potential - grad_entropy
    energy_total = quad_potential_particles(particles) - entropy_particles(particles, tree)
    return energy_total, grad_total


def run_experiment1(
        out_dir: str,
        max_iter: int
):
    # Sample initial configuration
    n_particles = 1000
    n_dimensions = 2
    particles = np.zeros((n_particles, n_dimensions))
    for i in range(n_particles // 2):
        particles[i] = sample_distribution(n_dimensions, -5, 0)
    for i in range(n_particles // 2, n_particles):
        particles[i] = sample_distribution(n_dimensions, 0, 5)
    draw_particles(particles, time=0.0)
    plt.savefig(f"{out_dir}/initial_configuration.pdf")

    # Optimize the system
    energies = []
    progress_bar = tqdm(range(max_iter))
    for i in progress_bar:
        if i % 25 == 0:
            draw_particles(particles, time=float(i) / max_iter)
            plt.savefig(f"{out_dir}/iteration_{format_iteration(i)}.png", dpi=300)
        energy_total, grad_total = total_energy_gradient(particles)
        particles -= 0.5 * grad_total
        energies.append(energy_total)
        progress_bar.set_description(f"Energy: {energy_total:.4f}")

    # Save the energies
    with open(f"{out_dir}/energies.csv", "w") as f:
        writer = csv.writer(f)
        for i, e in enumerate(energies):
            writer.writerow([i, e])

    draw_particles(particles, time=1.0)
    plt.savefig(f"{out_dir}/final_configuration.pdf")
