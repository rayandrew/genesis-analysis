---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Count the runs in a genesis trace

You'll read a sample of the genesis sweep with the dftracer-utils Python API, in pandas style, and see how many runs each application has on each system.
It takes about 10 minutes of your time.

Select a cell and press Shift+Enter to run it.
Every cell is safe to run again.

## Where this notebook runs

The trace is on corona's Lustre, so the notebook runs on corona.

```{code-cell} ipython3
!hostname
```

A login node such as `corona82` is fine for this sample.
Your host name can differ.

## Settings

The library reads its log level when it starts, so set it before the first import.
`warn` hides its startup line.

```{code-cell} ipython3
%env DFTRACER_UTILS_LOG_LEVEL=warn
```

The sample is the smallest part of the sweep, 162 MB.
A link to it goes into `data/` at the top of the repository, so the index that the first read writes lands in your folder and not next to the trace.

```{code-cell} ipython3
import os

TRACE = "/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/sample-s0-part0.pfw.gz"
SAMPLE = "../data/genesis-sample.pfw.gz"

os.makedirs("../data", exist_ok=True)
if not os.path.lexists(SAMPLE):
    os.symlink(TRACE, SAMPLE)
```

## Check what you need

```{code-cell} ipython3
import dftracer.utils

print("dftracer-utils", dftracer.utils.__version__)
if os.path.exists(SAMPLE):
    print(SAMPLE, os.path.getsize(SAMPLE), "bytes")
else:
    print(SAMPLE, "is missing: check that you can read", TRACE)
```

You need a sample of 161541455 bytes.

## 1. Read the runs

A genesis trace keeps one record for each run.
An [`Indexer`](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html) builds the trace's index and keeps those records as the `runs` table.

```{code-cell} ipython3
from dftracer.utils import Indexer

with Indexer(files=[SAMPLE]) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
runs.shape
```

That's 131 runs, each with 10 columns.
The index lands next to the link, in `data/.dftindex` at the top of the repository, and the next cells reuse it.

## 2. Look at a few runs

`runs` is a dftracer-utils `DataFrame`, which takes pandas spellings such as `df[["a", "b"]]` and `head()`, and shows as pandas does.

```{code-cell} ipython3
runs[["run", "app", "system", "unique_input", "nodes", "ppn", "papi_set"]].head()
```

A run is one application, one system, one input, one number of nodes, one number of processes per node (`ppn`) and one set of hardware counters (`papi_set`).
Every other record in the trace names its run by the `run` id.

## 3. Count the runs per application and system

`groupby(...).size()` counts the rows of each group, as in pandas.

```{code-cell} ipython3
runs.groupby(["app", "system"]).size()
```

The sample has amg on three systems and laghos on `matrix` only.

## 4. List the columns

```{code-cell} ipython3
runs.columns
```

## 5. Count the other records

Apart from the run records, the trace holds `func` records, one for each call path of a run, and `counter` records, one for each call path and counter.
A [`TraceViewer`](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html) reads them, and its `group_by` counts them without loading the records into memory.

```{code-cell} ipython3
from dftracer.utils import TraceViewer

tv = TraceViewer(SAMPLE)
tv.group_by("gtype").agg("count").sort_by("gtype").collect()
```

There are about 200 counter records for each call path, one for each counter.

## 6. Count the amg runs for each counter set

```{code-cell} ipython3
amg = runs[(runs["app"] == "amg") & (runs["system"] == "matrix")]
amg.groupby("papi_set").size()
```

The two sets have 12 and 15 runs.
Not every configuration ran with both counter sets, so compare runs of one set with runs of the same set.

## What you built

You read the runs of a genesis trace into a table and counted them and the other records with the dftracer-utils Python API.
The next notebook is [Find where amg spends its time](02-where-amg-spends-its-time.ipynb).

- How the genesis records are made: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
- The Python classes: [Python API Reference](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html)
