"""
make_gallery.py
----------------
Renders every clean canonical shape the sketch-completion model knows into a
single labelled PNG, so you can see the exact 16x16 renderings to draw toward.
The model completes a drawing to the nearest shape here -- if what you draw
doesn't resemble one of these, it has nothing to snap to.

Usage:
    python make_gallery.py   ->  writes shape_gallery.png next to this file
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(script_dir, "..", "data", "dataset.json")))

# One clean variant per label.
shapes = {}
for d in data:
    if "noisy" not in d["name"] and d["label"] not in shapes:
        shapes[d["label"]] = np.array(d["grid"], dtype=np.float32)

labels = sorted(shapes)
n = len(labels)
cols = 9
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7))
fig.suptitle(f"Shapes the model knows ({n} total) -- draw toward these", fontsize=14)

for ax in axes.flat:
    ax.axis("off")

for i, lbl in enumerate(labels):
    ax = axes.flat[i]
    ax.imshow(shapes[lbl], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(lbl, fontsize=7, pad=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(script_dir, "shape_gallery.png")
plt.savefig(out, dpi=110, facecolor="white")
print(f"wrote {out}")
