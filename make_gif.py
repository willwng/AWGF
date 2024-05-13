import os
import imageio
import numpy as np

from tqdm import tqdm


def get_key(filename):
    # file is named 3d_plot_{i}.png sort by i
    return int(filename.split('_')[-1].split('.')[0])


def main():
    png_dir = '.'
    images = []

    files = os.listdir(png_dir)
    files = [f for f in files if f.endswith('.png')]

    print("Reading images")
    for file_name in tqdm(sorted(files, key=get_key)):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.v2.imread(file_path))


    print("Creating gif")
    imageio.mimsave('movie.gif', images, duration=0.5)


if __name__ == "__main__":
    main()
