"""Count the runs of the full genesis sweep and read how amg's main scales.

Usage: python scripts/full_sweep.py <trace> <index dir> <amg input>
The first run builds the index in <index dir>; later runs reuse it.
"""

import sys
import time

from dftracer.utils import Indexer, TraceViewer, col

trace, index, amg_input = sys.argv[1], sys.argv[2], sys.argv[3]

start = time.time()
with Indexer(files=[trace], index_dir=index) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
print(f"index ready in {time.time() - start:.0f} s")
print(runs.groupby(["app", "system"]).size())

start = time.time()
tv = TraceViewer(trace, index)
main = tv.filter((col("gtype") == "func") & (col("name") == "main")).collect()
main = main[main["depth"] == 0].merge(runs, on="run")
amg = main[
    (main["app"] == "amg")
    & (main["system"] == "matrix")
    & (main["unique_input"] == amg_input)
]
amg = amg.assign(p50_s=amg["dur.p50"] / 1e6)
print(amg[["nodes", "ppn", "p50_s"]].groupby(["nodes", "ppn"]).median())
print(f"amg scaling in {time.time() - start:.0f} s")
