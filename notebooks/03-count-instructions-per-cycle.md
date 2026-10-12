---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Count instructions per cycle for each call path

You'll find the instructions per cycle (IPC) of each slow call path in one amg run, from its hardware counters.
It takes about 10 minutes of your time.

Select a cell and press Shift+Enter to run it.
This notebook runs on corona, after [Count the runs](01-count-the-runs.ipynb) made the link to the sample.

## Settings and check

```{code-cell} ipython3
%env DFTRACER_UTILS_LOG_LEVEL=warn
```

```{code-cell} ipython3
import os

SAMPLE = "../data/genesis-sample.pfw.gz"
RUN = "5400adb996f10a22"
print(SAMPLE, os.path.getsize(SAMPLE), "bytes")
```

The size must be 161541455 bytes.
`RUN` is the amg run with 8 nodes and 4 processes per node from [Find where amg spends its time](02-where-amg-spends-its-time.ipynb).

## 1. Load the counters of one run

A `counter` record holds one counter of one call path of one run.
Filter on the run inside the engine, so only that run's counters reach Python.

```{code-cell} ipython3
from dftracer.utils import TraceViewer, col

tv = TraceViewer(SAMPLE)
ctr = tv.filter((col("gtype") == "counter") & (col("run") == RUN)).collect()
ctr.shape
```

## 2. List its hardware counters

The PAPI counters have names that start with `PAPI_`, and `.str.startswith` works on a column as in pandas.

```{code-cell} ipython3
papi = ctr[ctr["metric"].str.startswith("PAPI_")]
papi.groupby("metric").size()
```

Each counter has a record for 434 call paths.
`PAPI_TOT_INS` counts instructions and `PAPI_TOT_CYC` counts cycles.

## 3. Read the two counters of main

`v.sum` is the counter's total over all calls of the path.

```{code-cell} ipython3
both = papi[papi["metric"].isin(["PAPI_TOT_INS", "PAPI_TOT_CYC"])]
both[both["path"] == "main"][["metric", "kind", "v.sum"]]
```

`main` ran about 2.2 instructions for each cycle.
Both are `delta` counters, so their sums add up over calls.

## 4. Put both counters on one row

`pivot_table` turns the values of `metric` into columns, one row for each `path`.

```{code-cell} ipython3
ipc = both.pivot_table(index="path", columns="metric", values="v.sum", aggfunc="sum")
ipc = ipc.assign(ipc=ipc["PAPI_TOT_INS"] / ipc["PAPI_TOT_CYC"])
ipc.nlargest(6, "PAPI_TOT_CYC")[["path", "ipc"]]
```

The paths come in order of the cycles they used.
The solve paths run at about 2.3 instructions per cycle.

## 5. Count branch misses

`PAPI_BR_MSP` counts mispredicted branches and `PAPI_BR_INS` all branches.
The same pivot gives the share of branches that miss.

```{code-cell} ipython3
br = papi[papi["metric"].isin(["PAPI_BR_MSP", "PAPI_BR_INS"])]
miss = br.pivot_table(index="path", columns="metric", values="v.sum", aggfunc="sum")
miss = miss.assign(miss=miss["PAPI_BR_MSP"] / miss["PAPI_BR_INS"])
miss[miss["path"].isin(["main;hypre_PCGSetup", "main;hypre_PCGSolve"])][["path", "miss"]]
```

The setup mispredicts about 3% of its branches and the solve under 0.5%.

## What you built

You loaded the hardware counters of one run and turned two of them into a ratio for each call path.

- How counters are attributed to calls: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
- The pandas spellings the engine takes: [Python API Reference](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html)
