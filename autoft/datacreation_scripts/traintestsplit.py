import os
import numpy as np
import shutil


data_path = "/iris/u/cchoi1/Data/caltech-101/101_ObjectCategories/"
categories = sorted(os.listdir(data_path))

os.makedirs("/iris/u/cchoi1/Data/caltech-101/train/", exist_ok=True)
os.makedirs("/iris/u/cchoi1/Data/caltech-101/val/", exist_ok=True)
os.makedirs("/iris/u/cchoi1/Data/caltech-101/test/", exist_ok=True)

for cat in categories:
    print(cat)
    os.makedirs(os.path.join("/iris/u/cchoi1/Data/caltech-101/train/", cat), exist_ok=True)
    os.makedirs(os.path.join("/iris/u/cchoi1/Data/caltech-101/val/", cat), exist_ok=True)
    os.makedirs(os.path.join("/iris/u/cchoi1/Data/caltech-101/test/", cat), exist_ok=True)
    
    image_files = os.listdir(os.path.join(data_path, cat))
    choices = np.random.choice([0, 1, 2], size=(len(image_files),), p=[.6, 0.2, 0.2])
    
    for (i,_f) in enumerate(image_files):
        if choices[i]==0:
            dest_path = os.path.join("/iris/u/cchoi1/Data/caltech-101/train/", cat, _f)
        if choices[i]==1:
            dest_path = os.path.join("/iris/u/cchoi1/Data/caltech-101/val/", cat, _f)
        if choices[i]==2:
            dest_path = os.path.join("/iris/u/cchoi1/Data/caltech-101/test/", cat, _f)
        
        origin_path = os.path.join(data_path, cat,  _f)
        shutil.copy(origin_path, dest_path)