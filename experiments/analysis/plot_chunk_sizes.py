"""Box plot of chunk size distribution per variable (log y)."""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

CSV_PATH = "/Users/fmr/development/bsc/ai4land-tfm/jobs-map/41548701/chunk_sizes.csv"
OUT_PATH = "/Users/fmr/development/bsc/ai4land-tfm/jobs-map/41548701/chunk_sizes.png"
EXCLUDE = {"time", "latitude", "longitude"}

sizes = defaultdict(list)

with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        if row["var"] in EXCLUDE:
            continue
        sizes[row["var"]].append(int(row["size_bytes"]))

vars_sorted = sorted(sizes)
data = [sizes[v] for v in vars_sorted]

fig, ax = plt.subplots(figsize=(14, 7))
ax.boxplot(data, tick_labels=vars_sorted, showfliers=True)
ax.set_yscale("log")
ax.set_ylabel("chunk size (bytes, log)")
ax.set_title("Compressed chunk size distribution per variable (n=100)")
ax.grid(True, which="both", axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(OUT_PATH)
