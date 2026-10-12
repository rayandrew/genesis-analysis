# Count instructions per cycle for each call path

You'll print the instructions per cycle (IPC) of each slow call path in one amg run, from its hardware counters.
It takes about 10 minutes of your time.

## Before you start

You need the Python session from [Find where amg spends its time](02-where-amg-spends-its-time.md), with `tv` in it.
If you closed it, start it again and open the trace.

```python
from dftracer.utils import TraceViewer, col

tv = TraceViewer("data/genesis-sample.pfw.gz")
print(tv.group_by("gtype").agg("count").sort_by("gtype").collect())
```

```text
     gtype    count
0  counter  4905256
1     func    23075
```

## 1. Load the counters of one run

A `counter` record holds one counter of one call path of one run.
Filter on the run inside the engine, so only that run's counters reach Python.

```python
ctr = tv.filter((col("gtype") == "counter") & (col("run") == "5400adb996f10a22")).collect()
print(ctr.shape)
```

```text
(79820, 18)
```

That's the run with 8 nodes and 4 processes per node from the last tutorial.

## 2. List its hardware counters

The PAPI counters have names that start with `PAPI_`.
`.str.startswith` works on a column as in pandas.

```python
papi = ctr[ctr["metric"].str.startswith("PAPI_")]
print(papi.groupby("metric").size())
```

```text
         metric  size
0    PAPI_BR_CN   434
1   PAPI_BR_INS   434
2   PAPI_BR_MSP   434
3   PAPI_BR_NTK   434
4   PAPI_BR_PRC   434
5   PAPI_BR_TKN   434
6   PAPI_BR_UCN   434
7  PAPI_REF_CYC   434
8  PAPI_TOT_CYC   434
9  PAPI_TOT_INS   434
```

Each counter has a record for 434 call paths.
`PAPI_TOT_INS` counts instructions and `PAPI_TOT_CYC` counts cycles.

## 3. Read the two counters of main

`v.sum` is the counter's total over all calls of the path.

```python
both = papi[papi["metric"].isin(["PAPI_TOT_INS", "PAPI_TOT_CYC"])]
print(both[both["path"] == "main"][["metric", "kind", "v.sum"]])
```

```text
         metric   kind         v.sum
0  PAPI_TOT_CYC  delta  1.508702e+11
1  PAPI_TOT_INS  delta  3.283328e+11
```

`main` ran about 2.2 instructions for each cycle.
Both are `delta` counters, so their sums add up over calls.

## 4. Put both counters on one row

`pivot_table` turns the values of `metric` into columns, one row for each `path`.

```python
ipc = both.pivot_table(index="path", columns="metric", values="v.sum", aggfunc="sum")
ipc = ipc.assign(ipc=ipc["PAPI_TOT_INS"] / ipc["PAPI_TOT_CYC"])
print(ipc.nlargest(6, "PAPI_TOT_CYC")[["path", "ipc"]])
```

```text
                                                                                                                 path       ipc
0                                                                                                                main  2.176260
1                                                                                                 main;hypre_PCGSolve  2.330656
2                                                                            main;hypre_PCGSolve;hypre_BoomerAMGSolve  2.320110
3                                                       main;hypre_PCGSolve;hypre_BoomerAMGSolve;hypre_BoomerAMGCycle  2.320115
4                                     main;hypre_PCGSolve;hypre_BoomerAMGSolve;hypre_BoomerAMGCycle;hypre_ParCSRRelax  2.318515
5  main;hypre_PCGSolve;hypre_BoomerAMGSolve;hypre_BoomerAMGCycle;hypre_ParCSRRelax;hypre_ParCSRMatrixMatvecOutOfPlace  2.319392
```

The paths come in order of the cycles they used.
The solve paths run at about 2.3 instructions per cycle.

## 5. Count branch misses

`PAPI_BR_MSP` counts mispredicted branches and `PAPI_BR_INS` all branches.
The same pivot gives the share of branches that miss.

```python
br = papi[papi["metric"].isin(["PAPI_BR_MSP", "PAPI_BR_INS"])]
miss = br.pivot_table(index="path", columns="metric", values="v.sum", aggfunc="sum")
miss = miss.assign(miss=miss["PAPI_BR_MSP"] / miss["PAPI_BR_INS"])
print(miss[miss["path"].isin(["main;hypre_PCGSetup", "main;hypre_PCGSolve"])][["path", "miss"]])
```

```text
                  path      miss
0  main;hypre_PCGSetup  0.029582
1  main;hypre_PCGSolve  0.004280
```

The setup mispredicts about 3% of its branches and the solve under 0.5%.

## What you built

You loaded the hardware counters of one run and turned two of them into a ratio for each call path.

- How counters are attributed to calls: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
- The pandas spellings the engine takes: [Python API Reference](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html)
