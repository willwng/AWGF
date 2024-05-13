# open cornell.png image and show it
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def main():
    img = mpimg.imread('cornell.png')
    # Convert to grayscale
    img = img[:, :, 0]
    # Threshold the image
    img[img < 0.7] = 0
    img[img >= 0.5] = 1
    # Get indices where the image is 1
    indices = np.where(img == 1)
    # Get the pixel locations where the image is 1
    locs = np.array(list(zip(indices[0], indices[1])))

    # Randomly sample N points
    n_points = 1000
    idx = np.random.choice(locs.shape[0], n_points, replace=False)
    points = locs[idx].astype(np.float32)
    # Scale to [-5, 5]
    height, width = img.shape
    points[:, 0] = 10.0 * points[:, 0] / float(height) - 5.0
    points[:, 1] = 10.0 * points[:, 1] / float(width) - 5.0
    # Rotate by 90 degrees
    points = np.flip(points, axis=1)
    plt.scatter(points[:, 1], points[:, 0], s=5)

    # Save the points with numpy
    np.save("cornell_points.npy", points)
    # draw image
    plt.imshow(img, cmap='gray')
    # no ticks
    plt.xticks([])
    plt.yticks([])

    plt.show()

    return


if __name__ == "__main__":
    main()
