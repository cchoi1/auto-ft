import pandas as pd
import h5py
from PIL import Image
import io
import numpy as np

# Read the CSV file in chunks
csv_file_path = '/home/carolinechoi/data/ImageNet/imagenet.csv'
chunksize = 10000  # Adjust based on your memory constraints

with h5py.File('/home/carolinechoi/data/ImageNet/imagenet.h5', 'w') as hdf:
    for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunksize)):
        # Process the chunk to images and labels
        images = []
        labels = []
        for _, row in chunk.iterrows():
            # Assuming the image is stored in a column named 'image_path'
            with open(row['filepath'], 'rb') as f:
                image_data = f.read()
            images.append(np.array(Image.open(io.BytesIO(image_data))))
            labels.append(row['label'])

        # Create datasets if not already created
        if i == 0:
            hdf.create_dataset('images', data=images, maxshape=(None,), chunks=True)
            hdf.create_dataset('labels', data=labels, maxshape=(None,), dtype='i8', chunks=True)
        else:
            hdf['images'].resize((hdf['images'].shape[0] + len(images)), axis=0)
            hdf['images'][-len(images):] = images
            hdf['labels'].resize((hdf['labels'].shape[0] + len(labels)), axis=0)
            hdf['labels'][-len(labels):] = labels
