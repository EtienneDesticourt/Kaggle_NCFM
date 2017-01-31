import os
import numpy as np
import shutil
import sys

np.random.seed(2016)

TRAIN_PATH = "train"
NEW_SAMPLE_PATH = "train_sample"
FISH_NAMES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Missing sampling ratio argument.")
    ratio = int(sys.argv[1])

    os.mkdir(NEW_SAMPLE_PATH)

    for fish_name in FISH_NAMES:
        # List images in fish directory
        path = os.path.join(TRAIN_PATH, fish_name)
        images = os.listdir(path)

        # Shuffle and take subsample
        np.random.shuffle(images)
        sample_size = len(images) // ratio
        image_subsample = images[:sample_size]

        # Write to new dir
        new_path = os.path.join(NEW_SAMPLE_PATH, fish_name)
        os.mkdir(new_path)
        for image in image_subsample:
            source = os.path.join(TRAIN_PATH, fish_name, image)
            dest = os.path.join(NEW_SAMPLE_PATH, fish_name, image)
            shutil.copy(source, dest)
