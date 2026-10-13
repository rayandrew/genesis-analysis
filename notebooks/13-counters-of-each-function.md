---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Read the counters of each function by layer

You'll build, for each C, C++ or Python function of each app on each system, the distribution of its duration and of each counter over all its calls, and put the functions side by side in one table.
It takes about 10 minutes of your time on a compute node.

## Before you start

Run the kernel inside an allocation and open the full sweep, as in [Measure the flop rate as processes share a node](06-flop-rate-as-processes-share-a-node.ipynb).

```{code-cell} ipython3
from dftracer.utils import DataFrame, Indexer, TraceViewer, col

TRACE = "/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz"
INDEX = "../data/full-index"
with Indexer(files=[TRACE], index_dir=INDEX) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
tv = TraceViewer(TRACE, INDEX)
print(runs.shape)
```

## 1. Give each function a layer

Each `func` record has the `cat` of its calls.
The categories of C and C++ functions, I/O, MPI and the GPU runtimes are known, and a Python annotation sets a category of its own, so every other category is Python.

```{code-cell} ipython3
KNOWN = {
    "C_APP": "C",
    "CPP_APP": "C++",
    "STDIO": "I/O",
    "POSIX": "I/O",
    "p2p": "MPI",
    "comm": "MPI",
    "collective": "MPI",
    "KERNEL_DISPATCH": "GPU",
    "MEMORY_COPY": "GPU",
    "PAGE_MIGRATION": "GPU",
    "SCRATCH_MEMORY": "GPU",
    "env": "other",
}


def layer_of(cat):
    if cat in KNOWN:
        return KNOWN[cat]
    if cat.startswith(("CUDA_", "HIP_")):
        return "GPU"
    return "Python"


func = tv.filter(col("gtype") == "func").collect()
cats = func.groupby("cat").size()["cat"].to_list()
layers = DataFrame.from_dict({"cat": cats, "layer": [layer_of(c) for c in cats]})
func = func.merge(layers, on="cat")
display(func.groupby("layer").size())
```

The table keeps the application functions, the C, C++ and Python layers.
This sweep has no Python layer, because its applications are C and C++.

## 2. Pick the counters

A counter means the same thing only on one CPU, so each row of the table is one function of one app on one system.
Each run measured one set of hardware counters, and the table takes each counter from the runs that measured it.

```{code-cell} ipython3
COUNTERS = ["PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_FP_OPS", "PAPI_L1_DCM", "PAPI_BR_MSP", "cpu.user_pct"]
APP = ["C", "C++", "Python"]

fns = func[func["layer"].isin(APP)].merge(runs[["run", "system", "app"]], on="run")
print(fns.shape)
```

These are the call paths of the C and C++ functions in all runs, each with the system and the app of its run.

## 3. Read the counters of the functions

A `counter` record holds the value credited to each call of one call path for one counter.
Read the counters in one scan, since the counter records are most of the file, and keep the ones of the application functions.

```{code-cell} ipython3
ctr = tv.filter((col("gtype") == "counter") & col("metric").isin(COUNTERS)).collect()
ctr = ctr.merge(fns[["run", "path", "system", "app", "layer", "cat"]], on=["run", "path"])
print(ctr.shape)
```

Each row is one counter of one call path in one run.

## 4. Fold the distribution of each counter

Fold the records of each function and counter: add the calls, take the least and the greatest value, weight the mean of each record by its calls and merge the sketches for the percentiles.
Fold the durations of the same functions the same way, so one table holds the duration and every counter.

```{code-cell} ipython3
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.float_format", "{:.4g}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

KEY = ["system", "app", "layer", "cat", "name"]
Q = {"p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9, "p99": 0.99}
STATS = ["min", "p25", "p50", "mean", "p75", "p90", "p99", "max"]

counters = ctr.groupby([*KEY, "metric"]).agg(
    calls=col("v.n").sum(),
    total=(col("v.avg") * col("v.n")).sum(),
    min=col("v.min").min(),
    **{name: col("v.sketch").sketch_quantile(q) for name, q in Q.items()},
    max=col("v.max").max(),
)
durations = fns.groupby(KEY).agg(
    calls=col("count").sum(),
    total=col("dur.sum").sum(),
    min=col("dur.min").min(),
    **{name: col("dur.sketch").sketch_quantile(q) for name, q in Q.items()},
    max=col("dur.max").max(),
)
dist = pd.concat([durations.to_pandas().assign(metric="dur"), counters.to_pandas()])
dist["mean"] = dist["total"] / dist["calls"]
one = dist[(dist["system"] == "tuolumne") & (dist["name"] == "hypre_CSRMatrixMatvecOutOfPlace")].set_index("metric")
display(one[["calls", *STATS]])
```

This is the distribution of one function on tuolumne, the matrix-vector product of amg: one row for its duration and one for each counter.
`dur` is in microseconds, the hardware counters count events in one call, and `cpu.user_pct` is the user time of the node in percent while the call ran.
`calls` is smaller for the hardware counters, because each run measured one set of them, so a counter comes only from the runs of its set.
The calls spread over four orders of magnitude, from a median of 686 microseconds to 0.35 s at the 99th percentile, and the counts per call spread with them.
The mean call takes 11 ms, 17 times the median, so a few long calls hold most of the time.

## 5. Put the functions side by side

`pivot_table` turns the long table into one row for each function and a group of columns for each counter, with every statistic in the group.
Show the duration, the instructions and the user time here.

```{code-cell} ipython3
wide = dist.pivot_table(index=KEY, columns="metric", values=STATS)
wide = wide.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([["dur", *COUNTERS], STATS]))
order = durations.to_pandas().sort_values(["system", "app", "layer", "total"], ascending=[True, True, True, False])
top = order.groupby(["system", "app", "layer"]).head(10).set_index(KEY).index
display(wide.loc[top, ["dur", "PAPI_TOT_INS", "cpu.user_pct"]])
```

Each app on each system shows its ten functions with the most time.
tuolumne has no laghos rows, since laghos didn't run there.
matrix has no value for `PAPI_L1_DCM`, since no counter set of matrix holds it.
The same function changes with the system: the median call of `Mult` in laghos takes 9 ms on matrix and 28 ms on tioga.
`wide` holds every counter, so pick others by name in the column list, such as `["PAPI_L1_DCM"]`.

## 6. Read the counters as rates

Counts per call grow with the length of the call, so divide them by the cycles to compare functions of different sizes.
Use the sums over all calls, so each rate counts every call once.

```{code-cell} ipython3
total = ctr.groupby([*KEY, "metric"]).agg(total=col("v.sum").sum())
total = total.pivot_table(values="total", index=KEY, columns="metric").to_pandas()
total = total.set_index(KEY)
rates = pd.DataFrame(
    {
        "ipc": total["PAPI_TOT_INS"] / total["PAPI_TOT_CYC"],
        "flops_per_cycle": total["PAPI_FP_OPS"] / total["PAPI_TOT_CYC"],
        "l1_miss_per_kinst": 1000 * total["PAPI_L1_DCM"] / total["PAPI_TOT_INS"],
        "br_miss_per_kinst": 1000 * total["PAPI_BR_MSP"] / total["PAPI_TOT_INS"],
    }
)
display(rates.loc[top].round(3))
```

The solve functions of amg in C, such as `hypre_GMRESSolve` and `hypre_ParCSRRelax`, run at 1.6 to 2 instructions per cycle on all three systems.
Their flop rate is 0.33 to 0.49 flops per cycle on matrix and tuolumne, and 0.1 to 0.18 on tioga.
The setup functions of amg, such as `hypre_GMRESSetup`, do almost no flops and miss 4 to 6 branches per thousand instructions on matrix and tuolumne.
laghos in C++ runs its solve at about 2.5 instructions per cycle on matrix and 2 on tioga, with almost no flops on the CPU.
minife in C++ runs at 0.3 to 0.6 instructions per cycle on matrix, about 0.9 on tuolumne and 0.8 to 2.6 on tioga.
On tuolumne minife offloads its solve to the GPU, so its CPU counters count the host side, which mostly waits.
Rows such as `hypre_BoomerAMGSolve` and `hypre_BoomerAMGCycle` match because one only calls the other.

## Limits

A counter is read per process and shared out among the calls by their time, so the rate of a short function is mostly the rate of the code around it.
Read the rates of long functions, such as a solve, as their own, and the rates of short ones as the rate of their phase.
A rate that needs two counters comes from the runs whose counter set holds both, so a system whose sets split them has no value there.
The records keep no sum of squares, so the table has no variance; the spread shows in the percentiles.

## What you built

You built one table of the counters of each application function, by system, app and layer, and turned the counts into rates.

- The same sketch merge in duql, `quantile(merge(v.sketch), q)`: [duql reference](https://dftracer.readthedocs.io/projects/utils/en/latest/reference/duql.html)
- Back to the start: [Count the runs in a genesis trace](01-count-the-runs.ipynb)
