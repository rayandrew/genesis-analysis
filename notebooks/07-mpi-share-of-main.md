---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure how much of main is MPI

You'll add up the time of the MPI calls of each run, divide it by the time of `main`, and draw that share as a heatmap over nodes and processes per node for each system.
It takes about 10 minutes of your time on a compute node.

## Before you start

Run the kernel inside an allocation and open the full sweep, as in [Measure the flop rate as processes share a node](06-flop-rate-as-processes-share-a-node.ipynb).

```{code-cell} ipython3
from dftracer.utils import Indexer, TraceViewer, col

TRACE = "/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz"
INDEX = "../data/full-index"
with Indexer(files=[TRACE], index_dir=INDEX) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
tv = TraceViewer(TRACE, INDEX)
print(runs.shape)
```

```{code-cell} ipython3
import os

import matplotlib.pyplot as plt
import numpy as np

os.makedirs("../figures", exist_ok=True)
```

## 1. Add up the MPI calls of each run

An MPI call has no traced children, so its `dur.sum` is all its own time.
Sum the `func` records whose name starts with `MPI_`, for each run.

```{code-cell} ipython3
calls = tv.filter((col("gtype") == "func") & col("name").str.startswith("MPI_")).collect()
mpi = calls.groupby("run")[["dur.sum"]].sum().rename(columns={"dur.sum": "mpi"})
main = tv.filter((col("gtype") == "func") & (col("path") == "main")).collect()
share = main[["run", "dur.sum"]].merge(mpi, on="run", how="left")
share = share.assign(mpi_share=share["mpi"].fillna(0) / share["dur.sum"])
share = share.merge(runs, on="run")
print(share.shape)
```

## 2. Read the share for one input

amg with `p1_n64x64x32` ran on every system, with the same problem size for each process.

```{code-cell} ipython3
amg = share[(share["app"] == "amg") & (share["unique_input"] == "p1_n64x64x32")]
heat = {}
for system in ["matrix", "tioga", "tuolumne"]:
    rows = amg[amg["system"] == system]
    heat[system] = rows.pivot_table(values="mpi_share", index="nodes", columns="ppn", aggfunc="median")
    print(system)
    print(heat[system])
```

Each value is the median over the runs of one setting, one for each counter set.

## 3. Draw the heatmaps

```{code-cell} ipython3
fig, axes = plt.subplots(1, len(heat), figsize=(4 * len(heat), 3.2), sharey=True)
for ax, (system, frame) in zip(axes, heat.items()):
    grid = frame.to_pandas().set_index("nodes")
    im = ax.imshow(grid.values, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("processes per node")
    ax.set_title(system)
    for (i, j), v in np.ndenumerate(grid.values):
        if v == v:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.5 else "black")
axes[0].set_ylabel("nodes")
fig.colorbar(im, ax=axes, label="MPI share of main")
fig.suptitle("amg p1_n64x64x32", y=1.06)
fig.savefig("../figures/mpi-share.png", dpi=120, bbox_inches="tight")
plt.show()
```


On matrix the share grows slowly, to about a quarter of `main` at the largest settings.
tuolumne reaches 0.3 to 0.4 from 16 processes per node, and 0.53 on 8 nodes with 4 processes per node.
On tioga it jumps from about 0.1 at 8 processes per node to about 0.7 at 16, on any number of nodes.
That's the setting where the flop rate of tioga fell on the previous page, so the processes spend the lost time waiting in MPI.

## 4. See which MPI calls take the time

```{code-cell} ipython3
at = calls.merge(amg[["run", "system", "nodes", "ppn"]], on="run")
tioga16 = at[(at["system"] == "tioga") & (at["nodes"] == 1) & (at["ppn"] == 16)]
print(tioga16.groupby("name")[["dur.sum"]].sum().nlargest(5, "dur.sum"))
```

`MPI_Waitall` takes most of it, then `MPI_Allreduce`, so the processes mostly wait for messages to finish.

## What you built

You computed the MPI share of `main` for every run and found the settings where communication takes most of the time.

- Next: [Compare the three systems](08-compare-the-systems.ipynb)
