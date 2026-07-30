"""Per-variable size distribution from chunk_sizes.csv."""

import csv
import statistics
from collections import defaultdict

CSV_PATH = "/Users/fmr/development/bsc/ai4land-tfm/jobs-map/41548701/chunk_sizes.csv"

sizes = defaultdict(list)

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sizes[row["var"]].append(int(row["size_bytes"]))

print("var,n,min,median,mean,p95,max,total")
for var in sorted(sizes):
    s = sorted(sizes[var])
    n = len(s)
    p95 = s[min(n - 1, int(n * 0.95))]
    print(f"{var},{n},{s[0]},{statistics.median(s):.0f},{statistics.mean(s):.0f},{p95},{s[-1]},{sum(s)}")
