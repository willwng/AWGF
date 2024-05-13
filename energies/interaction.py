import numpy as np


def interaction(p_i, p_j):
    """
    Interaction energy between two particles
    :param p_i: position of particle i
    :param p_j: position of particle j
    :return: interaction energy between particles i and j
    """
    return np.power(np.linalg.norm(p_i - p_j), 2)


def interaction_particles(particles: np.ndarray) -> float:
    """
    Lagrangian discretization of the interaction energy
    :param particles: list of positions of particles
    :return: interaction energy of the system
    """
    n = len(particles)
    interaction_energy = 0
    for i in range(n):
        for j in range(i + 1, n):
            interaction_energy += interaction(particles[i], particles[j])
    return (1 / (n ** 2)) * interaction_energy


def nabla_interaction_particles(particles: np.ndarray) -> np.ndarray:
    """
    Gradient of the interaction energy of the system
    :param particles: list of positions of particles
    :return: gradient of the interaction energy of the system
    """
    n = len(particles)
    grad_interaction = np.zeros_like(particles)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad_interaction[i] += 2 * (particles[i] - particles[j])
    return (1 / (n ** 2)) * grad_interaction
