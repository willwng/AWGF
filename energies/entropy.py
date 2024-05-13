import numpy as np


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
