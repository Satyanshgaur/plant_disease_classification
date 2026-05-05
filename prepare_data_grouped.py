import os, json, random, shutil
from collections import defaultdict

BASE = os.path.expanduser("~/plant_classification/PlantVillage-Dataset")

SOURCE = os.path.join(BASE, "raw/color")
LEAF_MAP = os.path.join(BASE, "leaf-map.json")

TRAIN = "data/train"
TEST = "data/test"

SPLIT = 0.8
from collections import defaultdict
import os

file_index = defaultdict(list)

for class_name in os.listdir(SOURCE):
    class_path = os.path.join(SOURCE, class_name)
    
    for fname in os.listdir(class_path):
        # extract identifier part after ___
        if "___" not in fname:
            continue
        
        key = fname.split("___")[1].rsplit(".", 1)[0].lower()
        file_index[(class_name, key)].append(fname)

with open(LEAF_MAP) as f:
    leaf_map = json.load(f)

leaf_groups = defaultdict(list)
for img_name, values in leaf_map.items():
    entry = values[0]
    class_name, leaf_id = entry.split(":::")
    
    key = img_name.lower()   # normalize
    
    leaf_groups[leaf_id].append((class_name, key))

leaf_ids = list(leaf_groups.keys())
random.shuffle(leaf_ids)

split_idx = int(len(leaf_ids) * SPLIT)

train_leafs = set(leaf_ids[:split_idx])
test_leafs = set(leaf_ids[split_idx:])
def copy_images(leaf_set, dest_root):
    for leaf_id in leaf_set:
        for class_name, key in leaf_groups[leaf_id]:
            
            matches = file_index.get((class_name, key), [])
            
            if not matches:
                continue  # or log
            
            for fname in matches:
                src = os.path.join(SOURCE, class_name, fname)
                dst = os.path.join(dest_root, class_name)
                
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, dst)

copy_images(train_leafs, TRAIN)
copy_images(test_leafs, TEST)
