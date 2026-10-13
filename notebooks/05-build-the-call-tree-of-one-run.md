---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Build the call tree of one amg run

You'll turn the call paths of one amg run into a call tree, find where its time goes with self time and a flame graph, and see which calls slow down when 16 processes share a node instead of 2.
It takes about 15 minutes of your time.

## Before you start

The first cell reads `runs` and opens the sample, as in [Find where amg spends its time](02-where-amg-spends-its-time.ipynb).

```{code-cell} ipython3
from dftracer.utils import Indexer, TraceViewer, col

with Indexer(files=["../data/genesis-sample.pfw.gz"]) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
tv = TraceViewer("../data/genesis-sample.pfw.gz")
print(runs.shape)
```

The plots show below each cell and go to `figures/` too.

```{code-cell} ipython3
import os

import matplotlib.pyplot as plt

os.makedirs("../figures", exist_ok=True)
```

## 1. Load the call paths of one run

Take amg with the input `p1_n64x64x32` on one matrix node, one run for each number of processes.

```{code-cell} ipython3
amg = runs[
    (runs["app"] == "amg")
    & (runs["system"] == "matrix")
    & (runs["unique_input"] == "p1_n64x64x32")
    & (runs["nodes"] == 1)
]
display(amg[["run", "ppn", "papi_set"]].sort_values("ppn"))
```

Load the call paths of the run with 16 processes.

```{code-cell} ipython3
RUN16 = amg[amg["ppn"] == 16]["run"].to_pandas().iloc[0]
func = tv.filter((col("gtype") == "func") & (col("run") == RUN16)).collect()
print(func.shape)
```

Each record is one call path, such as `main;hypre_PCGSolve`.
`parent` is the path one level up, and `dur.sum` is the time of all calls of the path, summed over the processes.
So `parent` already gives the edges of the call tree.

## 2. Compute the self time of each call path

`dur.sum` of a path includes the time of its children.
Its self time is what is left after the children's time is taken away.
Sum the children by `parent` and join that sum back to each path.

```{code-cell} ipython3
def self_time(func):
    children = func.groupby("parent")[["dur.sum"]].sum()
    children = children.rename(columns={"parent": "path", "dur.sum": "children"})
    tree = func.merge(children, on="path", how="left")
    tree = tree.assign(children=tree["children"].fillna(0))
    return tree.assign(self_s=(tree["dur.sum"] - tree["children"]) / 1e6)


tree = self_time(func)
display(tree.nlargest(5, "self_s")[["name", "depth", "count", "self_s"]])
```

A path with no children keeps all its time.
The self times add up to the time of the roots, the paths at depth 0, so nothing is counted twice.

```{code-cell} ipython3
roots = tree[tree["depth"] == 0]
display(roots[["name", "dur.sum"]])
print(round(tree["self_s"].sum(), 3), round(roots["dur.sum"].sum() / 1e6, 3))
```

Besides `main`, the run has three small roots, file reads that run outside `main`.

## 3. Sum the self time of each function

`hypre_CSRMatrixMatvecOutOfPlace` shows up at three depths above, called from three places.
Group the self time by `name` to see each function in one row.

```{code-cell} ipython3
by_name = tree.groupby("name")[["self_s"]].sum().nlargest(10, "self_s")
display(by_name)
```

The sparse matrix-vector product takes most of the time of this run.

```{code-cell} ipython3
top = by_name.to_pandas()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top["name"][::-1], top["self_s"][::-1])
ax.set_xlabel("self time summed over 16 processes (s)")
ax.set_title("amg p1_n64x64x32, 1 node, 16 processes")
fig.tight_layout()
fig.savefig("../figures/self-time-16.png", dpi=120)
plt.show()
```


## 4. Draw the flame graph

A flame graph puts each call path in a box as wide as its time, with its children above it, widest first.
Place each child after its siblings inside its parent, starting from `main`.

```{code-cell} ipython3
import zlib


def flame(tree, title, path):
    t = tree[["path", "parent", "name", "depth", "dur.sum"]].to_pandas()
    t["w"] = t["dur.sum"] / 1e6
    kids = {p: g.sort_values("w", ascending=False) for p, g in t.groupby("parent")}
    left = {}

    def place(p, x):
        left[p] = x
        if p not in kids:
            return
        for child, w in zip(kids[p]["path"], kids[p]["w"]):
            place(child, x)
            x += w

    place("main", 0.0)
    t = t[t["path"].isin(left)]
    width = t["w"].max()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.get_cmap("tab20")
    for p, name, depth, w in zip(t["path"], t["name"], t["depth"], t["w"]):
        ax.barh(depth, w, left=left[p], height=0.9, edgecolor="white", linewidth=0.3,
                color=colors(zlib.crc32(name.encode()) % 20))
        if w > 0.009 * width * len(name):
            ax.text(left[p] + w / 2, depth, name, ha="center", va="center", fontsize=7)
    ax.set_xlim(0, width)
    ax.set_xlabel("time summed over the processes (s)")
    ax.set_ylabel("depth")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.show()


flame(tree, "amg p1_n64x64x32, 1 node, 16 processes", "../figures/flame-16.png")
```


`hypre_PCGSolve` takes about 70% of `main`, and most of it sits in the stack of `hypre_BoomerAMGCycle`, `hypre_ParCSRRelax` and the matrix-vector product at the top.
`hypre_PCGSetup` and `BuildIJLaplacian27pt`, the box at the right with no room for its name, fill most of the rest.

## 5. Compare 2 and 16 processes

Each process of these runs gets the same problem size, so the time per process would stay flat if the processes didn't get in each other's way.
Compute the self time per function and per process for the runs with 2 and 16 processes, and join them.

```{code-cell} ipython3
def per_process(ppn):
    run = amg[amg["ppn"] == ppn]["run"].to_pandas().iloc[0]
    f = tv.filter((col("gtype") == "func") & (col("run") == run)).collect()
    by_name = self_time(f).groupby("name")[["self_s"]].sum()
    return by_name.assign(**{f"p{ppn}": by_name["self_s"] / ppn})[["name", f"p{ppn}"]]


both = per_process(2).merge(per_process(16), on="name")
both = both.assign(grow=both["p16"] - both["p2"])
display(both.nlargest(6, "grow"))
```

```{code-cell} ipython3
top = both.nlargest(8, "p16").to_pandas()[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([i + 0.2 for i in range(len(top))], top["p2"], height=0.4, label="2 processes")
ax.barh([i - 0.2 for i in range(len(top))], top["p16"], height=0.4, label="16 processes")
ax.set_yticks(range(len(top)), top["name"])
ax.set_xlabel("self time per process (s)")
ax.set_title("amg p1_n64x64x32 on one matrix node")
ax.legend()
fig.tight_layout()
fig.savefig("../figures/self-2-vs-16.png", dpi=120)
plt.show()
```


The matrix-vector product takes 2.4 times as long per process with 16 processes, 0.45 s against 1.08 s, which is about two thirds of the growth of `main`.
It reads the whole matrix for little arithmetic, so memory bandwidth limits it, and 16 processes on one node share that bandwidth.
The MPI calls grow too, from almost nothing to about 0.05 s each per process.

## What you built

You built the call tree of a run from its `path` and `parent` columns, split its time into self time, drew it as a flame graph and found the calls that stop scaling on a full node.

- What the `func` fields mean: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
- The pandas spellings the engine takes: [Python API Reference](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html)
