import pandas as pd
import h5py
from PIL import Image
import numpy as np
import os

csv_file_path = '/home/carolinechoi/data/ImageNet/imagenet.csv'
hdf5_file_path = '/home/carolinechoi/data/ImageNet/imagenet.h5'
chunksize = 10000  # Adjust based on your memory constraints

def process_image(filepath):
    with Image.open(filepath) as img:
        # Optionally, resize or preprocess the image as needed
        return np.array(img)

with h5py.File(hdf5_file_path, 'w') as hdf:
    for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunksize, sep='\t')):
        images, filepaths, captions, labels = [], [], [], []
        for _, row in chunk.iterrows():
            filepath = row['filepath']
            filepaths.append(filepath.encode('utf-8'))  # Convert filepath to bytes for HDF5
            captions.append(row['title'].encode('utf-8'))  # Convert caption to bytes for HDF5
            labels.append(row['label'])
            filepath = row['filepath']
            if os.path.exists(filepath):
                images.append(process_image(filepath))
                # Other processing remains the same
            else:
                print(f"File not found: {filepath}")

            # if os.path.exists(filepath):
            #     with Image.open(filepath) as img:
            #         images.append(np.array(img))
            # else:
            #     print(f"File not found: {filepath}")

        # Convert to numpy arrays and handle dtype for HDF5
        # images_np = np.array(images, dtype=object)
        labels_np = np.array(labels, dtype=np.int64)
        print('len filepaths', len(filepaths))
        print('len captions', len(captions))
        print('len labels', len(labels))

        # Write data to HDF5, resizing arrays as necessary
        if i == 0:
            hdf.create_dataset('filepath', data=filepaths, maxshape=(None,), chunks=True)
            # hdf.create_dataset('images', data=images_np, maxshape=(None,),
            #                    dtype=h5py.special_dtype(vlen=np.dtype('uint8')), chunks=True)
            hdf.create_dataset('title', data=captions, maxshape=(None,), chunks=True)
            hdf.create_dataset('label', data=labels_np, maxshape=(None,), dtype='i8', chunks=True)
        else:
            hdf['filepath'].resize((hdf['filepath'].shape[0] + len(filepaths)), axis=0)
            hdf['filepath'][-len(filepaths):] = filepaths
            # hdf['images'].resize((hdf['images'].shape[0] + len(images)), axis=0)
            # hdf['images'][-len(images):] = images_np
            hdf['title'].resize((hdf['title'].shape[0] + len(captions)), axis=0)
            hdf['title'][-len(captions):] = captions
            hdf['label'].resize((hdf['label'].shape[0] + len(labels)), axis=0)
            hdf['label'][-len(labels):] = labels_np

