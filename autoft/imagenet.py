import os
import shutil
import scipy.io

# Paths
val_dir = "/iris/u/cchoi1/Data/ILSVRC2012/val"
ground_truth_file = "/iris/u/cchoi1/Data/ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
meta_mat_file = "/iris/u/cchoi1/Data/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.mat"

# Load ground truth
with open(ground_truth_file, 'r') as f:
    labels = [int(line.strip()) for line in f]

# Load wnids from meta.mat
meta_data = scipy.io.loadmat(meta_mat_file)
wnids = [synset[0][1][0] for synset in meta_data['synsets']]

print(f"Number of labels: {len(labels)}")
print(f"Number of wnids: {len(wnids)}")
print(f"Min label value: {min(labels)}")
print(f"Max label value: {max(labels)}")

# Ensure wnids directories exist
for wnid in wnids:
    os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)

# Move images
for idx, filename in enumerate(sorted(os.listdir(val_dir))):
    # Skip directories
    if os.path.isdir(os.path.join(val_dir, filename)):
        continue

    src = os.path.join(val_dir, filename)
    dst = os.path.join(val_dir, wnids[labels[idx] - 1], filename)
    shutil.move(src, dst)
