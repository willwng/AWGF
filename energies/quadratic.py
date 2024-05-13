import numpy as np


def quad_potential(x: np.ndarray) -> float:
    """
    Potential energy of a single particle
    :param x: position of the particle
    :return: potential energy of the particle
    """
    return np.square(np.linalg.norm(x))


def d_quad_potential(x: np.ndarray) -> np.ndarray:
    """
    Gradient of the potential energy of a single particle
    :param x: position of the particle
    :return: gradient of the potential energy of the particle
    """
    return 2 * x


def quad_potential_particles(particles: np.ndarray) -> float:
    """
    Lagrangian discretization of the potential energy
    :param particles: list of positions of particles
    :return: potential energy of the system
    """
    n = len(particles)
    return (1 / n) * np.sum(np.vectorize(quad_potential)(particles))


def nabla_quad_potential_particles(particles: np.ndarray) -> np.ndarray:
    """
    Gradient of the potential energy of the system
    :param particles: list of positions of particles
    :return: gradient of the potential energy of the system
    """
    n = len(particles)
    return (1 / n) * np.vectorize(d_quad_potential)(particles)
