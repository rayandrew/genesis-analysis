# Find where amg spends its time

You'll print how long amg's `main` takes as the number of processes grows, and which calls inside `main` take that time.
It takes about 15 minutes of your time.

## Before you start

You need the Python session from [Count the runs](01-count-the-runs.md), with `runs` and `tv` in it.
If you closed it, start it again and read both.

```python
from dftracer.utils import Indexer, TraceViewer

with Indexer(files=["data/genesis-sample.pfw.gz"]) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
tv = TraceViewer("data/genesis-sample.pfw.gz")
print(runs.shape)
```

```text
(131, 10)
```

## 1. Load the call paths

A `func` record holds the duration of one call path of one run, over all its processes.
`filter` with a `col` condition picks records inside the engine, so only the `func` records reach Python.

```python
from dftracer.utils import col

func = tv.filter(col("gtype") == "func").collect()
print(func.shape)
```

```text
(23075, 19)
```

23,075 call paths, out of almost 5 million records in the sample.

## 2. Read one call path

Look at the `main` of three runs, sorted by run id so the same three show each time.

```python
main = func[(func["name"] == "main") & (func["depth"] == 0)]
print(main[["run", "path", "count", "dur.p50", "dur.max"]].sort_values("run").head(3))
```

```text
                run  path  count       dur.p50       dur.max
0  00408f6a9c4994b2  main      4  1.567006e+08  1.567006e+08
1  010d8a3fb3f5751d  main      2  9.097600e+04  9.869900e+04
2  021f52cc43d47c4e  main      4  1.056776e+08  1.057050e+08
```

`count` is the number of calls, one for each process here.
`dur.p50` is the median call duration and `dur.max` the longest one, both in microseconds.

## 3. Add the run settings

A `func` record names its run by id only.
`merge` adds the columns of `runs` to each record with the same `run`, as in pandas.

```python
main = main.merge(runs, on="run")
amg = main[(main["app"] == "amg") & (main["system"] == "matrix") & (main["unique_input"] == "p1_n64x64x32")]
amg = amg.assign(p50_s=amg["dur.p50"] / 1e6)
print(amg[["nodes", "ppn", "papi_set", "count", "p50_s"]].sort_values(["nodes", "ppn", "papi_set"]))
```

```text
    nodes  ppn                papi_set  count     p50_s
0       1    2          papi_set2_flop      2  0.838144
1       1    4  papi_set1_branch_cycle      4  0.977782
2       1    8          papi_set2_flop      8  1.264006
3       1   16  papi_set1_branch_cycle     16  1.813067
4       2    1  papi_set1_branch_cycle      2  0.900321
5       2    1          papi_set2_flop      2  0.882493
6       2    2          papi_set2_flop      4  0.937065
7       2    8          papi_set2_flop     16  1.316543
8       4    1  papi_set1_branch_cycle      4  0.975309
9       4    2          papi_set2_flop      8  1.056543
10      4    8  papi_set1_branch_cycle     32  1.397956
11      8    1          papi_set2_flop      8  1.077887
12      8    2          papi_set2_flop     16  1.144542
13      8    4  papi_set1_branch_cycle     32  1.426197
14      8    4          papi_set2_flop     32  1.290473
```

Each row is one run.
Two settings show up twice, such as 2 nodes with 1 process per node, because they ran once for each counter set.

## 4. Take the median of each setting

The two runs of one setting are the same program, so group them and take the median.
Use the method `.median()`, which is exact.
`agg({"p50_s": "median"})` is refused, because the engine's `agg` median is an estimate from a sketch.

```python
scaling = amg[["nodes", "ppn", "p50_s"]].groupby(["nodes", "ppn"]).median()
print(scaling)
```

```text
    nodes  ppn     p50_s
0       1    2  0.838144
1       1    4  0.977782
2       1    8  1.264006
3       1   16  1.813067
4       2    1  0.891407
5       2    2  0.937065
6       2    8  1.316543
7       4    1  0.975309
8       4    2  1.056543
9       4    8  1.397956
10      8    1  1.077887
11      8    2  1.144542
12      8    4  1.358335
```

One row for each setting now, with the median of the runs' medians.
On one node, `main` goes from 0.84 s with 2 processes to 1.81 s with 16.
Each process gets the same problem size here, so the extra time is the cost of more processes on one node.

## 5. Pick one run

To see inside `main`, take one run.
Find the run with 8 nodes and 4 processes per node that has the cycle counters.

```python
pick = runs[(runs["app"] == "amg") & (runs["system"] == "matrix") & (runs["unique_input"] == "p1_n64x64x32") & (runs["nodes"] == 8) & (runs["ppn"] == 4)]
print(pick[["run", "papi_set"]])
```

```text
                run                papi_set
0  c1b917c40593a9c1          papi_set2_flop
1  5400adb996f10a22  papi_set1_branch_cycle
```

The run with `papi_set1_branch_cycle` is `5400adb996f10a22`.
[Count instructions per cycle](03-count-instructions-per-cycle.md) uses the same run.

## 6. Find the slow calls inside main

The calls directly inside `main` have `depth` 1.
`nlargest` keeps the rows with the largest total time.

```python
inside = func[(func["run"] == "5400adb996f10a22") & (func["depth"] == 1)].nlargest(5, "dur.sum")
inside = inside.assign(total_s=inside["dur.sum"] / 1e6, p50_ms=inside["dur.p50"] / 1e3)
print(inside[["path", "count", "total_s", "p50_ms"]])
```

```text
                          path  count    total_s      p50_ms
0          main;hypre_PCGSolve     32  25.719681  803.734000
1          main;hypre_PCGSetup     32  12.411026  387.855000
2    main;BuildIJLaplacian27pt     32   6.323746  217.609988
3       main;hypre_PrintTiming     32   0.213410    6.838641
4  main;hypre_BoomerAMGDestroy     32   0.015332    0.487971
```

`hypre_PCGSolve` takes the most time, and `hypre_PCGSetup` comes second.
`total_s` adds up all processes, so it's larger than the wall time of the run.

## 7. Compare a smaller input

The input `p1_n32x32x16` is 8 times smaller.
Take the same medians for it.

```python
small = main[(main["app"] == "amg") & (main["system"] == "matrix") & (main["unique_input"] == "p1_n32x32x16")]
small = small.assign(p50_s=small["dur.p50"] / 1e6)
print(small[["nodes", "ppn", "p50_s"]].groupby(["nodes", "ppn"]).median())
```

```text
   nodes  ppn     p50_s
0      1    1  0.071310
1      1    8  0.151820
2      1   16  0.282227
3      2    1  0.090976
4      2    2  0.101766
5      2    8  0.167787
6      4    1  0.107246
7      4    2  0.134651
8      4    4  0.265791
9      8    1  0.134651
```

`main` takes about 0.28 s with 16 processes on one node, against 1.81 s for the larger input.
The time drops by more than 6 times while the work drops by 8 times, so part of the time doesn't shrink with the input.

## What you built

You loaded the call paths of a trace, joined each one to its run settings and grouped the runs of one setting.
Keep the session open for [Count instructions per cycle](03-count-instructions-per-cycle.md).

- What the `func` fields mean: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
- The pandas spellings the engine takes: [Python API Reference](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html)
