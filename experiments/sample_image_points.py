"""
This script loads an image, thresholds it, and randomly samples N points from the image.
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

in_file = "heart.png"
out_file = "heart_points.npy"

def main():
    # Open and convert to grayscale
    img = Image.open(in_file).convert("L")
    img = Image.eval(img, lambda x: 255 - x)
    # Threshold the image
    img = np.array(img)
    # Threshold the image
    threshold = 100
    img[img < threshold] = 0
    img[img >= threshold] = 1
    # Get indices where the image is 1
    indices = np.where(img > 0)
    # Get the pixel locations where the image is 1
    locs = np.array(list(zip(indices[0], indices[1])))

    # Randomly sample N points
    n_points = 5000
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
    np.save(out_file, points)
    # draw image
    # img = np.rot90(img)
    # plt.imshow(img)
    # no ticks
    plt.xticks([])
    plt.yticks([])

    plt.show()

    return


if __name__ == "__main__":
    main()
