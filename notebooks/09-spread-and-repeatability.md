---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Check how repeatable the runs are

You'll measure two kinds of spread in the sweep, between the processes of one run and between the runs of one setting.
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

os.makedirs("../figures", exist_ok=True)
main = tv.filter((col("gtype") == "func") & (col("path") == "main")).collect().merge(runs, on="run")
main = main.assign(t=main["dur.p50"] / 1e6)
```

## 1. Spread between the processes of one run

Each process calls `main` once, so `dur.p50` and `dur.p99` of `main` are the median and the slowest processes.
Their ratio is 1 when all processes take the same time.

```{code-cell} ipython3
main = main.assign(spread=main["dur.p99"] / main["dur.p50"])
print(main.groupby(["app", "system"])[["spread"]].quantile(0.99))
```

For 99% of the runs of laghos, and of amg on tioga and tuolumne, the slowest process is within about 16% of the median one.
amg on matrix and minife on tioga have runs where it is 1.5 to 2.6 times slower.

## 2. Spread between the runs of one setting

A setting ran once for each counter set, so it has 2 to 9 runs.
They are the same program with the same input, so they should take about the same time.

```{code-cell} ipython3
setting = ["app", "system", "unique_input", "nodes", "ppn"]
g = main.groupby(setting)[["t"]]
rep = g.min().rename(columns={"t": "fastest"}).merge(g.max().rename(columns={"t": "slowest"}), on=setting)
rep = rep.assign(ratio=rep["slowest"] / rep["fastest"])
print(rep.groupby(["app", "system"])[["ratio"]].median())
print(rep.nlargest(5, "ratio"))
```

```{code-cell} ipython3
r = rep.to_pandas()
fig, ax = plt.subplots(figsize=(7, 4))
for (app, system), rows in r.groupby(["app", "system"]):
    if len(rows) > 20:
        ax.ecdf(rows["ratio"], label=f"{app} {system}")
ax.set_xscale("log")
ax.set_xlabel("slowest run / fastest run of one setting")
ax.set_ylabel("share of the settings")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("../figures/run-to-run.png", dpi=120)
plt.show()
```


For amg on matrix and tioga and for laghos, the runs of one setting are within a few percent of each other in the median.
minife differs more, by about 36% on tioga.
On tuolumne a few amg settings have runs that are up to 192 times slower than the fastest one.

## 3. Look at one slow setting

```{code-cell} ipython3
worst = rep.nlargest(1, "ratio").to_pandas().iloc[0]
same = main[
    (main["app"] == worst["app"])
    & (main["system"] == worst["system"])
    & (main["unique_input"] == worst["unique_input"])
    & (main["nodes"] == worst["nodes"])
    & (main["ppn"] == worst["ppn"])
]
print(same[["run", "papi_set", "t"]].sort_values("t"))
```

Over all such settings the slow runs come from at least five different counter sets, so no one counter set makes them slow.
[Read the host counters of each run](12-host-counters.ipynb) shows that the host CPU counters don't explain them either.

## What you built

You measured the spread between processes and between runs, and found the settings whose runs disagree.

- Next: [Read merged distributions](10-merged-distributions.ipynb)
