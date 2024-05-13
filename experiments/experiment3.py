"""
Experiment 3: Accelerated Gradient Descent with Steering
"""

import csv

from tqdm import tqdm

from energies.energy_combiner import total_energy_gradient_quad, get_image_kl
from helper.distribution_helper import sample_distribution
from helper.plot_tools import *


def run_experiment3(
        out_dir: str,
        max_iter: int
):
    # Sample initial configuration
    n_particles = 10000
    n_dimensions = 2
    particles = np.zeros((n_particles, n_dimensions))
    for i in range(n_particles // 2):
        particles[i] = sample_distribution(n_dimensions, -5, 5)
    for i in range(n_particles // 2, n_particles):
        particles[i] = sample_distribution(n_dimensions, 5, 5)
    draw_particles(particles, time=0.0)
    plt.savefig(f"{out_dir}/initial_configuration.pdf")

    # Optimize the system
    energies = []
    progress_bar = tqdm(range(max_iter))
    velocity = np.zeros_like(particles)
    gamma = 0.5
    eta = 0.5
    for i in progress_bar:
        if i % 25 == 0:
            draw_particles(particles, time=float(i) / max_iter)
            plt.savefig(f"{out_dir}/iteration_{format_iteration(i)}.png", dpi=300)
        energy_total, grad_total = get_image_kl(particles)
        expected_vsq = np.sum(velocity ** 2) / n_particles
        expected_gsq = np.sum(grad_total ** 2) / n_particles
        tau_factor = 1 + (gamma * np.sqrt(expected_vsq / expected_gsq))
        velocity -= eta * (gamma * velocity + tau_factor * grad_total)
        particles += eta * velocity
        energies.append(energy_total)
        progress_bar.set_description(f"Energy: {energy_total:.4f}")

    # Save the energies
    with open(f"{out_dir}/energies.csv", "w") as f:
        writer = csv.writer(f)
        for i, e in enumerate(energies):
            writer.writerow([i, e])

    draw_particles(particles, time=1.0)
    plt.savefig(f"{out_dir}/final_configuration.pdf")
