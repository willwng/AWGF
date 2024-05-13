from typing import Tuple

import numpy as np
import scipy

from energies.entropy import nabla_entropy_particles, entropy_particles
from energies.divergence import kl_divergence_cornell, nabla_kl_divergence_cornell
from energies.quadratic import nabla_quad_potential_particles, quad_potential_particles


def total_energy_gradient_quad_entropy(particles: np.ndarray) -> Tuple[float, np.ndarray]:
    tree = scipy.spatial.cKDTree(particles)
    grad_potential = nabla_quad_potential_particles(particles)
    grad_entropy = nabla_entropy_particles(particles, tree)
    grad_total = grad_potential - grad_entropy
    energy_total = quad_potential_particles(particles) - entropy_particles(particles, tree)
    return energy_total, grad_total


def total_energy_gradient_quad(particles: np.ndarray) -> Tuple[float, np.ndarray]:
    grad_potential = nabla_quad_potential_particles(particles)
    grad_total = grad_potential
    energy_total = quad_potential_particles(particles)
    return energy_total, grad_total


def get_cornell_kl(particles: np.ndarray) -> Tuple[float, np.ndarray]:
    energy_total = kl_divergence_cornell(particles)
    grad_total = nabla_kl_divergence_cornell(particles)
    return energy_total, grad_total
