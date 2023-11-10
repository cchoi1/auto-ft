import pandas as pd
import h5py
from PIL import Image
import numpy as np
import os

csv_file_path = '/home/carolinechoi/data/ImageNet/imagenet.csv'
hdf5_file_path = '/home/carolinechoi/data/ImageNet/imagenet.h5'
chunksize = 10000  # Adjust based on your memory constraints

with h5py.File(hdf5_file_path, 'w') as hdf:
    for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunksize)):
        images, filepaths, captions, labels = [], [], [], []
        for _, row in chunk.iterrows():
            filepath = row['filepath']
            filepaths.append(filepath.encode('utf-8'))  # Convert filepath to bytes for HDF5
            captions.append(row['title'].encode('utf-8'))  # Convert caption to bytes for HDF5
            labels.append(row['label'])

            if os.path.exists(filepath):
                with Image.open(filepath) as img:
                    images.append(np.array(img))
            else:
                print(f"File not found: {filepath}")

        # Convert to numpy arrays and handle dtype for HDF5
        images_np = np.array(images, dtype=object)
        labels_np = np.array(labels, dtype=np.int64)

        # Write data to HDF5, resizing arrays as necessary
        if i == 0:
            hdf.create_dataset('images', data=images_np, maxshape=(None,), chunks=True, dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
            hdf.create_dataset('filepaths', data=filepaths, maxshape=(None,), chunks=True)
            hdf.create_dataset('captions', data=captions, maxshape=(None,), chunks=True)
            hdf.create_dataset('labels', data=labels_np, maxshape=(None,), dtype='i8', chunks=True)
        else:
            hdf['images'].resize((hdf['images'].shape[0] + len(images)), axis=0)
            hdf['images'][-len(images):] = images_np
            hdf['filepaths'].resize((hdf['filepaths'].shape[0] + len(filepaths)), axis=0)
            hdf['filepaths'][-len(filepaths):] = filepaths
            hdf['captions'].resize((hdf['captions'].shape[0] + len(captions)), axis=0)
            hdf['captions'][-len(captions):] = captions
            hdf['labels'].resize((hdf['labels'].shape[0] + len(labels)), axis=0)
            hdf['labels'][-len(labels):] = labels_np

