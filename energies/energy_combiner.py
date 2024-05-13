from typing import Tuple

import numpy as np
import scipy

from energies.entropy import nabla_entropy_particles, entropy_particles
from energies.quadratic import nabla_quad_potential_particles, quad_potential_particles


def total_energy_gradient(particles: np.ndarray) -> Tuple[float, np.ndarray]:
    tree = scipy.spatial.cKDTree(particles)
    grad_potential = nabla_quad_potential_particles(particles)
    grad_entropy = nabla_entropy_particles(particles, tree)
    grad_total = grad_potential - grad_entropy
    energy_total = quad_potential_particles(particles) - entropy_particles(particles, tree)
    return energy_total, grad_total
