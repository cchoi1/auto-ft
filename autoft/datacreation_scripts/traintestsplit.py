import argparse
import os
import shutil

import numpy as np


def main(args):
    categories = sorted(os.listdir(args.data_path))

    # Create directories for train, validation, and test sets
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    for cat in categories:
        print(cat)
        os.makedirs(os.path.join(args.train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(args.val_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(args.test_dir, cat), exist_ok=True)

        image_files = os.listdir(os.path.join(args.data_path, cat))
        choices = np.random.choice([0, 1, 2], size=(len(image_files),), p=[.6, 0.2, 0.2])

        for (i, _f) in enumerate(image_files):
            if choices[i] == 0:
                dest_path = os.path.join(args.train_dir, cat, _f)
            elif choices[i] == 1:
                dest_path = os.path.join(args.val_dir, cat, _f)
            else:
                dest_path = os.path.join(args.test_dir, cat, _f)

            origin_path = os.path.join(args.data_path, cat, _f)
            shutil.copy(origin_path, dest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caltech 101 Dataset Preparation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the Caltech 101 dataset')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory to save training images')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory to save validation images')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory to save test images')
    args = parser.parse_args()

    main(args)