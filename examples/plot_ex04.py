import h5py
import numpy as np
import matplotlib.pyplot as plt

# Read VTKHDF
f = h5py.File("output/ex04_cantilever.vtkhdf", "r")

# Steps
steps = list(f.keys())
print(f"Steps: {steps}")

# Use last step
last = steps[-1]
g = f[last]

# Coords (original)
x = g["points"][:, 0]
y = g["points"][:, 1]

# Connectivity
conn = g["connectivity"][:]  # flat, 4 per element
offsets = g["offsets"][:]
elem_offset = np.diff(np.concatenate([[0], offsets]))

# Solution - need to find the displacement dataset
print("Datasets in step:", list(g.keys()))
print("Points shape:", g["points"].shape)

# Try to find displacement data
# The exporter may store different field names
for key in g.keys():
    print(f"  {key}: shape={g[key].shape if hasattr(g[key], 'shape') else '?'}")

# The displacement might be stored in a specific field
# Check if there are point data fields
if "point_data" in g:
    pd = g["point_data"]
    print("Point data fields:", list(pd.keys()))
    for k in pd.keys():
        print(f"  {k}: shape={pd[k].shape}")
        data = pd[k][:]
        print(f"    range: [{data.min():.4f}, {data.max():.4f}]")

# Also check for field_data or cell_data
for k in ["field_data", "cell_data"]:
    if k in g:
        print(f"{k}:", list(g[k].keys()))

f.close()
