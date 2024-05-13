import numpy as np
import scipy

# Initialize the image points and the KDTree
image_points = np.load("experiments/heart_points.npy")
image_tree = scipy.spatial.cKDTree(image_points)


def kl_divergence(particles: np.ndarray, q_tree, m: int) -> float:
    """
    Estimate the KL divergence between discretizations of p and q
        based on F. Perez-Cruz, "Kullback-Leibler divergence estimation of continuous distributions," 2008
    :param particles: np.ndarray, shape (n, d)
    :param q_tree: tree for q distribution
    :param m: number of points in q
    """
    p_tree = scipy.spatial.cKDTree(particles)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = p_tree.query(particles, k=2, p=2)[0][:, 1]
    s = q_tree.query(particles, k=1, p=2)[0]

    n, d = particles.shape
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))


def nabla_kl_divergence(particles: np.ndarray, q_tree, m: int) -> np.ndarray:
    p_tree = scipy.spatial.cKDTree(particles)
    # The Euclidean distances to the kth nearest-neighbour
    r_nn = p_tree.query(particles, k=2, p=2)[1][:, 1]
    s_nn = q_tree.query(particles, k=1, p=2)[1]
    r_pos = p_tree.data[r_nn]
    s_pos = q_tree.data[s_nn]
    r = np.linalg.norm(particles - r_pos, axis=1)
    s = np.linalg.norm(particles - s_pos, axis=1)
    n, d = particles.shape
    grad = (-d / n) * ((particles - r_pos) / r[:, None]) + (d / m) * ((particles - s_pos) / s[:, None])
    # replace nan values with 0
    grad = np.nan_to_num(grad)
    return grad


def kl_divergence_image(p: np.ndarray) -> float:
    return kl_divergence(particles=p, q_tree=image_tree, m=image_points.shape[0])


def nabla_kl_divergence_image(p: np.ndarray) -> np.ndarray:
    return nabla_kl_divergence(particles=p, q_tree=image_tree, m=image_points.shape[0])
