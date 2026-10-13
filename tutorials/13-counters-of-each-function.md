# Read the counters of each function by layer

You'll build, for each C, C++ or Python function of each app on each system, the distribution of its duration and of each counter over all its calls, and put the functions side by side in one table.
It takes about 10 minutes of your time on a compute node.

## Before you start

Start Python on a compute node and open the full sweep, as in [Measure the flop rate as processes share a node](06-flop-rate-as-processes-share-a-node.md).

```python
from dftracer.utils import DataFrame, Indexer, TraceViewer, col

TRACE = "/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz"
INDEX = "data/full-index"
with Indexer(files=[TRACE], index_dir=INDEX) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
tv = TraceViewer(TRACE, INDEX)
print(runs.shape)
```

```text
(9624, 10)
```

## 1. Give each function a layer

Each `func` record has the `cat` of its calls.
The categories of C and C++ functions, I/O, MPI and the GPU runtimes are known, and a Python annotation sets a category of its own, so every other category is Python.

```python
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
print(func.groupby("layer").size())
```

```text
   layer     size
0      C   432291
1    C++    98430
2    GPU   398041
3    I/O    86339
4    MPI  1272155
5  other     4806
```

The table keeps the application functions, the C, C++ and Python layers.
This sweep has no Python layer, because its applications are C and C++.

## 2. Pick the counters

A counter means the same thing only on one CPU, so each row of the table is one function of one app on one system.
Each run measured one set of hardware counters, and the table takes each counter from the runs that measured it.

```python
COUNTERS = ["PAPI_TOT_INS", "PAPI_TOT_CYC", "PAPI_FP_OPS", "PAPI_L1_DCM", "PAPI_BR_MSP", "cpu.user_pct"]
APP = ["C", "C++", "Python"]

fns = func[func["layer"].isin(APP)].merge(runs[["run", "system", "app"]], on="run")
print(fns.shape)
```

```text
(530721, 22)
```

These are the call paths of the C and C++ functions in all runs, each with the system and the app of its run.

## 3. Read the counters of the functions

A `counter` record holds the value credited to each call of one call path for one counter.
Read the counters in one scan, since the counter records are most of the file, and keep the ones of the application functions.

```python
ctr = tv.filter((col("gtype") == "counter") & col("metric").isin(COUNTERS)).collect()
ctr = ctr.merge(fns[["run", "path", "system", "app", "layer", "cat"]], on=["run", "path"])
print(ctr.shape)
```

```text
(955316, 22)
```

Each row is one counter of one call path in one run.

## 4. Fold the distribution of each counter

Fold the records of each function and counter: add the calls, take the least and the greatest value, weight the mean of each record by its calls and merge the sketches for the percentiles.
Fold the durations of the same functions the same way, so one table holds the duration and every counter.

```python
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
print(one[["calls", *STATS]].to_string())
```

```text
                 calls       min       p25       p50      mean       p75       p90       p99       max
metric                                                                                                
dur           45438467       100       308     685.6 1.145e+04      4317 2.509e+04 3.517e+05 1.906e+06
PAPI_BR_MSP    5037419    0.8799      1686      4584 5.241e+04 2.014e+04  7.39e+04 7.371e+05 7.685e+06
PAPI_FP_OPS    5049982     4.857 3.057e+05 8.146e+05  2.05e+07 5.557e+06  4.36e+07 8.415e+08 9.124e+08
PAPI_L1_DCM    5048621     10.66 5.698e+04  1.43e+05 3.641e+06  9.56e+05 7.653e+06 1.507e+08  1.78e+08
PAPI_TOT_CYC   5036862 7.742e+04 1.122e+06 2.547e+06 4.203e+07 1.572e+07 9.324e+07 1.307e+09 1.679e+09
PAPI_TOT_INS   5036862 1.001e+05 2.351e+06 4.831e+06 7.572e+07 2.923e+07 1.699e+08 2.579e+09 2.945e+09
cpu.user_pct  45426048    0.1653     5.755     8.936     10.19     14.16     18.73     23.81     46.73
```

This is the distribution of one function on tuolumne, the matrix-vector product of amg: one row for its duration and one for each counter.
`dur` is in microseconds, the hardware counters count events in one call, and `cpu.user_pct` is the user time of the node in percent while the call ran.
`calls` is smaller for the hardware counters, because each run measured one set of them, so a counter comes only from the runs of its set.
The calls spread over four orders of magnitude, from a median of 686 microseconds to 0.35 s at the 99th percentile, and the counts per call spread with them.
The mean call takes 11 ms, 17 times the median, so a few long calls hold most of the time.

## 5. Put the functions side by side

`pivot_table` turns the long table into one row for each function and a group of columns for each counter, with every statistic in the group.
Show the duration, the instructions and the user time here.

```python
wide = dist.pivot_table(index=KEY, columns="metric", values=STATS)
wide = wide.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([["dur", *COUNTERS], STATS]))
order = durations.to_pandas().sort_values(["system", "app", "layer", "total"], ascending=[True, True, True, False])
top = order.groupby(["system", "app", "layer"]).head(10).set_index(KEY).index
print(wide.loc[top, ["dur", "PAPI_TOT_INS", "cpu.user_pct"]].to_string())
```

```text
                                                                       dur                                                                       PAPI_TOT_INS                                                                       cpu.user_pct                                               
                                                                       min       p25       p50      mean       p75       p90       p99       max          min       p25       p50      mean       p75       p90       p99       max          min     p25    p50   mean    p75   p90   p99   max
system   app    layer cat     name                                                                                                                                                                                                                                                             
matrix   amg    C     C_APP   main                                    3636 4.382e+05 1.078e+06 1.689e+07 3.651e+06 2.698e+07 2.075e+08 1.614e+09    5.447e+05 3.482e+09 1.025e+10 1.081e+11 3.916e+10 9.826e+10 1.297e+12 9.927e+12       0.2244   4.438  17.29  15.94  23.34 30.27 35.52 37.66
                              hypre_GMRESSolve                         645      5712 1.751e+04 9.136e+05 2.311e+05 1.426e+06 2.039e+07 1.407e+08    1.697e+04  6.25e+07 1.878e+08 6.013e+09 1.444e+09 8.564e+09 1.353e+11 9.915e+11       0.2218   8.936  20.29   17.2  23.34 30.27 35.52 37.66
                              hypre_ParCSRMatrixMatvecOutOfPlace       100     206.5     788.6 1.447e+04      7558 2.887e+04 4.941e+05 8.206e+05            0 1.545e+06 5.339e+06 8.733e+07 3.716e+07 1.185e+08 3.482e+09 7.027e+09       0.2216   4.015  9.681  14.65  22.42 30.27 35.52 37.66
                              hypre_BoomerAMGSolve                     113     837.4      1755 8.808e+04 3.387e+04 1.678e+05  3.05e+06 4.424e+06    1.962e+04 9.347e+06 2.165e+07 5.621e+08 2.075e+08 1.255e+09 2.065e+10 3.217e+10       0.2216   8.416  19.49  16.91  23.34 30.27 35.52 37.66
                              hypre_BoomerAMGCycle                     106     820.8      1755 8.807e+04 3.387e+04 1.678e+05  3.05e+06 4.424e+06            0 9.162e+06 2.165e+07  5.62e+08 2.075e+08 1.255e+09 2.065e+10 3.217e+10       0.2216   8.416  19.49  16.91  23.34 30.27 35.52 37.66
                              hypre_CSRMatrixMatvecOutOfPlace          100     354.3      1003 1.302e+04      6977 2.665e+04 4.382e+05 7.722e+05            0 2.399e+06 5.447e+06 7.581e+07 3.362e+07 1.072e+08 3.279e+09 3.785e+09       0.2216   3.222  9.117  14.31  22.42 29.67 35.52 35.65
                              hypre_ParCSRRelax                        100     179.5       728 1.495e+04      5712 3.065e+04 5.353e+05 9.256e+05         5915 1.317e+06 4.735e+06 8.869e+07 3.642e+07  1.03e+08 3.697e+09 7.317e+09       0.2216   4.015   13.6  14.89  22.42 30.27 35.52 37.66
                              hypre_GMRESSetup                        1038 4.573e+04 7.691e+04 1.178e+06 6.157e+05 2.086e+06 4.106e+07 4.632e+07    2.097e+05 4.618e+08 8.085e+08 7.405e+09 4.338e+09 1.759e+10  1.72e+11 3.238e+11       0.2215   8.936  20.29   17.2  23.34 30.27 35.52 37.66
                              hypre_PCGSolve                          1597 3.255e+04 1.347e+05 1.987e+06 1.514e+06 6.141e+06  2.08e+07 2.098e+07     1.27e+06 4.618e+08  1.36e+09 1.094e+10 7.749e+09 4.595e+10 5.613e+10 3.325e+11       0.2587   3.354  16.28  14.98  23.34 30.27 35.52 35.65
                              hypre_BoomerAMGCoarsenPMIS               100      2144      3985 4.097e+04      6977  2.46e+04 7.371e+05 6.949e+06    2.708e+04  1.92e+07 3.791e+07 2.711e+08 6.907e+07 1.994e+08 5.299e+09 3.046e+10       0.2215   7.926  18.73  16.54  23.34 30.27 35.52 37.66
         laghos C++   CPP_APP main                               2.943e+06 4.819e+07 1.284e+08 1.605e+08 2.293e+08 3.561e+08  5.99e+08 7.439e+08    1.622e+10 4.065e+11 1.105e+12 1.391e+12 1.974e+12 3.065e+12 5.366e+12 7.649e+12        1.079   22.42  28.51  27.56  34.13 42.53 47.95 48.69
                              Mult                                    2203      8025      9048 1.056e+04 1.083e+04 1.751e+04 2.411e+04 2.734e+06    1.616e+06 6.907e+07 7.945e+07 9.309e+07   9.9e+07 1.537e+08 2.248e+08  2.32e+10        1.079   23.81  29.67  28.62  34.82 41.69 47.95 48.74
                              SolveVelocity                           2139      7558      8694 1.011e+04 1.041e+04 1.682e+04 2.363e+04 2.734e+06    1.117e+06 6.636e+07 7.634e+07  8.91e+07 9.512e+07 1.507e+08  2.16e+08 2.319e+10        1.079   23.81  29.67  28.62  34.82 41.69 47.95 48.74
                              SolveEnergy                              100     354.3     391.6       436     441.5     518.1      1065 2.058e+06    1.189e+05 2.989e+06 3.439e+06 3.856e+06 3.955e+06 4.735e+06 9.729e+06 1.213e+10        1.079   23.81  29.67  28.62  34.82 41.69 47.95 48.74
                              GetTimeStepEstimate                      100     108.9     127.8     832.5     179.5     407.6      1901 6.367e+05    2.604e+05  9.56e+05 1.122e+06 9.179e+06 1.608e+06 3.651e+06 1.738e+07 5.028e+09        1.079   23.34  27.94  27.25  32.79 36.97    47 48.74
                              LagrangianHydroOperator            1.169e+04 1.377e+04 1.462e+04 2.756e+04 1.935e+04 8.332e+04 1.294e+05 2.521e+05    2.583e+07 1.185e+08 1.363e+08 2.381e+08 1.768e+08  5.99e+08 1.387e+09 1.741e+09        1.079   22.42  28.51  27.58  34.13 42.53 47.95 48.74
                              UpdateQuadratureData                     100     108.9     179.5     319.4     340.4     550.2      1791 2.603e+05    2.609e+05 8.311e+05 1.576e+06 2.696e+06 3.111e+06 4.831e+06 1.703e+07 1.639e+09        1.079   23.81  29.67  28.97  34.82 42.53 47.95 48.74
                              Rho0DetJ0Vol                            5066      5712      5945      8357      6188      6440  5.93e+04 1.082e+05    2.112e+06 4.819e+07 5.433e+07  6.53e+07 6.005e+07 7.047e+07 3.561e+08 5.718e+08        1.079   22.42  28.51  27.58  34.13 42.53 47.95 48.74
                              InternalEnergy                           100     925.4      1065      3640      1300      2466 6.554e+04 1.232e+05    1.842e+05 7.501e+06 9.162e+06 3.306e+07 1.165e+07  2.08e+07 6.489e+08 1.405e+09        1.079   22.42  28.51  27.52  34.13 42.53 47.95 48.74
                              SetEssentialTrueDofs                     100     242.3     267.8     724.5     376.2      3263      6065  4.53e+04    1.318e+05 2.004e+06 2.399e+06 6.186e+06 3.304e+06 2.209e+07 4.916e+07 4.053e+08        1.079   22.88  29.08  28.16  34.82 42.53 47.95 48.74
         minife C++   CPP_APP driver                             3.674e+04 9.584e+04 1.171e+05 1.427e+05 1.746e+05 2.311e+05 3.887e+05 9.873e+05     4.71e+05 2.117e+08 8.085e+08 8.533e+08 1.255e+09 1.729e+09 3.027e+09 5.664e+09      0.08429 0.09729 0.1188 0.2921 0.1738 1.335 1.335 1.335
                              cg_solve                            2.21e+04 3.065e+04 3.526e+04 4.703e+04 4.665e+04   8.5e+04 1.818e+05 4.478e+05    9.685e+04 1.954e+08 3.421e+08 3.738e+08 4.618e+08 6.489e+08 1.416e+09 4.197e+09      0.08429 0.09729 0.1188 0.2921 0.1738 1.335 1.335 1.335
                              make_local_matrix                          1 3.004e+04 4.137e+04 4.503e+04  5.26e+04 6.424e+04 1.374e+05  1.95e+05         2934  2.16e+08 4.096e+08 4.352e+08 5.755e+08 7.614e+08 1.695e+09 2.282e+09      0.08429 0.09729 0.1188 0.2921 0.1738 1.335 1.335 1.335
                              dot                                       19     30.88     36.24     50.28        47     77.49     232.8 7.118e+04            0 2.879e+05  3.81e+05 5.307e+05 5.143e+05 8.825e+05 2.759e+06 1.112e+09      0.08429 0.09729 0.1188 0.2925 0.1738 1.335 1.335 1.335
                              generate_matrix_structure                930      4677      8522 1.956e+04 1.859e+04 5.053e+04  1.43e+05  3.04e+05    1.495e+06 4.723e+07 1.094e+08  1.98e+08 2.435e+08 5.003e+08 1.255e+09 1.817e+09      0.08419 0.09729 0.1188 0.2921 0.1738 1.335 1.335 1.335
                              operator()                                10     36.97     43.39     56.87     49.91     59.75     301.9 1.313e+05            0 3.447e+05 4.382e+05 6.318e+05 5.461e+05  6.67e+05 3.651e+06 1.724e+09      0.08429 0.09729 0.1188 0.2925 0.1738 1.335 1.335 1.335
                              finish_exchange_externals                  2     19.11     24.78     38.79     32.14     41.69     284.4 1.312e+05            0 1.854e+05 2.554e+05 4.545e+05 3.447e+05 4.653e+05 3.439e+06 1.723e+09      0.08429 0.09729 0.1188 0.2925 0.1738 1.335 1.335 1.335
                              assemble_FE_data                         530      1003      1557      3468      3263      7262 2.945e+04  1.43e+05    2.897e+05 1.212e+07 2.393e+07 4.907e+07 5.543e+07 1.185e+08 3.561e+08 6.624e+08      0.08419 0.09729 0.1188 0.2921 0.1738 1.335 1.335 1.335
                              perform_element_loop_cuda                522      1003      1526      3456      3198      7118 2.945e+04  1.43e+05    1.783e+05 1.212e+07 2.393e+07 4.894e+07 5.543e+07 1.185e+08 3.561e+08 6.622e+08      0.08419 0.09729 0.1188 0.2921 0.1738 1.335 1.335 1.335
                              waxpby                                     2     2.975     2.975     3.231     2.975     4.015      5.99       125            0  2.46e+04 3.065e+04 3.157e+04  3.67e+04 4.306e+04 6.297e+04 1.181e+06      0.08429 0.09729 0.1188 0.2925 0.1738 1.335 1.335 1.335
tioga    amg    C     C_APP   main                                    6362 2.122e+07 3.791e+07 1.141e+08 7.334e+07 2.435e+08 1.049e+09 1.072e+09    9.366e+06 8.542e+10 3.073e+11 6.708e+11 5.066e+11 9.417e+11 6.821e+12 7.711e+12        2.211   2.638  2.801  2.956  3.287 3.561 4.179 4.229
                              hypre_GMRESSolve                        1889  4.46e+06 1.097e+07 1.485e+07 1.703e+07  3.57e+07 7.634e+07 8.576e+07    1.407e+07 2.024e+10 4.328e+10 6.442e+10 7.279e+10 1.496e+11 3.464e+11 5.471e+11        2.581   3.222  3.354  3.265  3.422 3.491 3.986 3.986
                              hypre_BoomerAMGSolve                     140  4.21e+05 7.225e+05 1.122e+06 2.004e+06 2.259e+06 2.651e+06 3.397e+06    4.028e+05 2.154e+09 5.627e+09 6.115e+09 8.737e+09 1.203e+10 1.759e+10   2.9e+10        2.211   2.746  3.287  3.133  3.422 3.491 4.179 4.231
                              hypre_BoomerAMGCycle                     132  4.21e+05 7.225e+05 1.122e+06 2.004e+06 2.259e+06 2.651e+06 3.397e+06    3.906e+05 2.154e+09 5.627e+09 6.114e+09 8.737e+09 1.203e+10 1.759e+10   2.9e+10        2.211   2.746  3.287  3.133  3.422 3.491 4.179 4.231
                              hypre_ParCSRMatrixMatvecOutOfPlace       100 1.377e+04  2.56e+04 4.837e+04 3.127e+04 1.645e+05 3.312e+05  1.09e+06    1.073e+05 5.769e+07 2.117e+08 2.403e+08 3.287e+08 4.263e+08 1.503e+09 1.216e+10        2.211   2.746  3.287   3.13  3.422 3.491 4.179 4.231
                              hypre_ParCSRRelax                        100 1.323e+04 2.509e+04 5.092e+04 3.065e+04 1.818e+05  3.66e+05  1.09e+06    1.167e+05 5.543e+07  2.16e+08 2.439e+08 3.287e+08 4.437e+08 1.533e+09  1.22e+10        2.211   2.746  3.287  3.132  3.422 3.491 4.179 4.231
                              hypre_CSRMatrixMatvecOutOfPlace          100     320.6      1108 3.071e+04 1.462e+04 1.401e+05 2.766e+05  5.77e+05    3.983e+04 1.925e+06 7.353e+06 7.206e+07 5.016e+07 3.158e+08 5.871e+08 1.108e+09        2.211   2.746  3.354  3.169  3.422 3.561 4.015 4.231
                              hypre_GMRESSetup                        4501 8.458e+06 3.295e+07 3.064e+07 4.819e+07 5.543e+07 6.907e+07 7.438e+07    6.226e+06 2.625e+10 1.224e+11 1.931e+11 3.464e+11 4.493e+11 5.712e+11 6.063e+11        2.581   3.222  3.354  3.264  3.422 3.491 3.986 3.986
                              hypre_NewCommPkgCreate_core              136 1.243e+05 5.461e+05  7.37e+05 1.215e+06 1.641e+06 2.872e+06 5.252e+06    2.965e+05 8.085e+08 3.345e+09 5.177e+09 8.564e+09 1.278e+10 2.149e+10 3.792e+10        2.211   2.691  3.158  3.062  3.422 3.561 4.179 4.229
                              hypre_DataExchangeList                   100 5.053e+04 2.357e+05 3.688e+05 5.353e+05  9.56e+05 1.742e+06 4.263e+06    1.223e+05 2.974e+08 1.416e+09 2.591e+09 3.697e+09 7.153e+09 1.499e+10 3.052e+10        2.211   2.691  3.158  3.062  3.422 3.561 4.179 4.229
         laghos C++   CPP_APP main                               1.398e+07  1.31e+08 4.179e+08  5.15e+08 7.316e+08 1.136e+09 1.799e+09 1.805e+09    3.943e+10 5.168e+11 2.612e+12 3.407e+12 5.259e+12 8.331e+12 1.518e+13 1.653e+13      0.03032   3.422   4.35  4.865   5.99 7.769  10.7 12.05
                              Mult                                    5077 2.316e+04 2.773e+04 5.203e+04 4.665e+04  6.05e+04 2.712e+05 9.511e+06    8.985e+06  1.03e+08 1.537e+08 3.489e+08 2.293e+08 3.935e+08 3.027e+09 2.775e+10      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              SolveVelocity                           4947 2.226e+04 2.665e+04 4.989e+04 4.482e+04 5.813e+04 2.658e+05  9.51e+06    7.956e+06   9.9e+07 1.448e+08 3.352e+08  2.16e+08 3.633e+08 2.967e+09 2.774e+10      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              SolveEnergy                              114      1086      1224      2110      1381      1557      1864 1.425e+06    3.889e+05 4.831e+06  6.02e+06 1.353e+07 7.501e+06 9.536e+06 1.703e+07 5.446e+09      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              UpdateQuadratureData                       0    0.9901     111.1     98.09     156.1     183.1     214.9  1.16e+06            0      3985 5.041e+05 4.859e+05 7.371e+05 9.753e+05 1.742e+06 1.743e+09      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              GetTimeStepEstimate                       89     165.7     190.6     345.3     214.9     242.3     424.2 6.029e+05    7.827e+04 7.225e+05 9.371e+05 2.568e+06 1.168e+06 1.576e+06 2.759e+06 3.779e+09      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              SetEssentialTrueDofs                       0     8.936     10.08     10.42     10.91     12.06     26.85 7.135e+05            0 3.896e+04 4.759e+04 5.575e+04 5.698e+04   7.1e+04 1.374e+05 5.265e+08      0.03019   3.422   4.35  4.826  6.111 7.464  10.7 12.06
                              ForceMultTranspose                         5     8.936     10.08     10.12     10.91     12.06     19.11 2.315e+05         6259 3.896e+04 4.665e+04 5.157e+04 5.585e+04 7.243e+04 1.347e+05 2.009e+07      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              ForceMult                                  5     8.936     8.936     10.03     10.08     12.06     15.96 3.445e+05         5547 3.819e+04 4.573e+04 5.313e+04 5.474e+04 6.959e+04 1.194e+05 8.303e+07      0.03019   3.422  4.438  4.892  6.361 7.615 11.14 12.06
                              LagrangianHydroOperator             3.54e+04 3.896e+04 4.953e+04 7.277e+04 8.167e+04  1.32e+05  3.81e+05 1.402e+06    5.317e+08 7.925e+08 1.091e+09 1.358e+09 1.503e+09 2.794e+09 5.856e+09 6.061e+09      0.03019   3.422   4.35  4.867   5.99 7.769  10.7 12.06
         minife C++   CPP_APP main                               1.584e+05  7.52e+05 1.036e+06 1.297e+06 1.545e+06 2.399e+06 5.901e+06 1.081e+07    8.303e+08 5.975e+09 1.499e+10 2.065e+10 2.522e+10 4.076e+10 1.043e+11 2.422e+11        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              driver                             1.101e+05 2.265e+05 4.747e+05 8.615e+05 1.015e+06 2.044e+06 5.784e+06 1.079e+07    8.108e+08 4.086e+09 1.005e+10 1.729e+10 2.149e+10 3.838e+10 1.043e+11 2.422e+11        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              generate_matrix_structure               3739 4.759e+04 1.374e+05     3e+05 3.965e+05 8.146e+05 1.455e+06 2.687e+06    1.247e+07 8.249e+08 3.279e+09 4.208e+09 6.095e+09 9.277e+09 1.624e+10 3.084e+10        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              cg_solve                            8.72e+04  1.43e+05 1.892e+05 2.494e+05 2.311e+05 3.447e+05  1.24e+06 5.585e+06    8.081e+08 2.288e+09 3.027e+09 3.387e+09 4.005e+09  4.99e+09 1.179e+10 1.753e+10        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              init_matrix                             2752 2.773e+04 1.194e+05 2.429e+05 3.246e+05 7.082e+05  1.29e+06 2.545e+06    1.204e+07  6.62e+08 2.631e+09 3.287e+09  4.99e+09 7.445e+09 1.278e+10 2.377e+10        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              assemble_FE_data                        1037 1.433e+04 5.366e+04  1.23e+05 1.374e+05 3.119e+05  9.56e+05 2.846e+06    7.093e+06 3.287e+08 1.911e+09 6.067e+09 6.603e+09 1.624e+10 5.181e+10 1.433e+11        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              perform_element_loop                    1033 1.433e+04 5.366e+04  1.23e+05 1.374e+05 3.119e+05  9.56e+05 2.846e+06    7.076e+06 3.287e+08 1.911e+09 6.067e+09 6.603e+09 1.624e+10 5.181e+10 1.433e+11        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              operator()                               100     214.9       308     583.3       488     982.7      5273   6.5e+04    1.034e+06 3.439e+06 4.929e+06 8.087e+06 7.653e+06 1.394e+07 4.106e+07 5.457e+08        7.852   8.249  9.301  10.43  13.33 13.88 14.04 14.04
                              make_local_matrix                        168 1.297e+04 4.393e+04 9.183e+04 1.171e+05  2.22e+05 6.538e+05 1.282e+06    3.002e+06 2.746e+08 9.116e+08 1.627e+09 2.478e+09 4.253e+09 7.298e+09 1.101e+10        7.852   8.249  9.301  10.42  13.33 13.88 14.04 14.04
                              impose_dirichlet                         100      1720  1.15e+04 4.447e+04 4.137e+04 1.171e+05 5.247e+05  1.79e+06    1.429e+06 3.642e+07  3.49e+08 1.247e+09 1.255e+09 3.345e+09 1.357e+10 3.327e+10        7.852   8.249  9.301  10.45  13.33 13.88 14.04 14.04
tuolumne amg    C     C_APP   main                                1.15e+04 2.766e+05 6.281e+05  1.72e+07 3.111e+06 2.393e+07 2.858e+08 1.618e+09    9.755e+05 1.695e+09   4.7e+09 1.073e+11 3.142e+10 1.686e+11 1.492e+12 7.049e+12       0.1653   6.111  10.28  10.85  15.34 19.89 26.32 46.73
                              hypre_GMRESSolve                         825     1e+04 2.887e+04 1.038e+06 2.265e+05 1.813e+06 1.959e+07 1.096e+08    5.439e+05 8.781e+07 2.293e+08 7.188e+09 1.911e+09 1.278e+10   1.3e+11 6.023e+11       0.1653   5.641  10.49  10.61  15.03 19.11 24.78 46.73
                              hypre_ParCSRMatrixMatvecOutOfPlace       100     135.7     290.1      9784      1409 1.682e+04 1.782e+05 2.001e+06    3.111e+04 1.168e+06 2.497e+06   6.9e+07 1.212e+07 1.568e+08 2.112e+09 5.008e+09       0.1653   5.871  9.681  10.51  14.73 19.11 27.94 46.73
                              hypre_BoomerAMGSolve                     126      1557      2952 9.385e+04 2.945e+04 2.405e+05 2.351e+06 1.038e+07    3.009e+04  1.34e+07 2.865e+07 6.926e+08 1.768e+08 1.949e+09  1.69e+10 2.007e+10       0.1653   5.755  10.28  10.66  15.03 19.11 24.78 46.73
                              hypre_BoomerAMGCycle                     120      1526      2894 9.384e+04 2.945e+04 2.405e+05 2.351e+06 1.038e+07            0  1.34e+07 2.865e+07 6.925e+08 1.768e+08 1.949e+09  1.69e+10 2.007e+10       0.1653   5.755  10.28  10.66  15.03 19.11 24.78 46.73
                              hypre_CSRMatrixMatvecOutOfPlace          100       308     685.6 1.145e+04      4317 2.509e+04 3.517e+05 1.906e+06    1.001e+05 2.351e+06 4.831e+06 7.572e+07 2.923e+07 1.699e+08 2.579e+09 2.945e+09       0.1653   5.755  8.936  10.19  14.16 18.73 23.81 46.73
                              hypre_ParCSRRelax                        100       133     247.2      9911      1224 1.323e+04 1.782e+05 2.034e+06         9261 1.122e+06 2.214e+06 6.843e+07 9.162e+06  1.31e+08 1.661e+09 5.228e+09       0.1653   5.871  9.681  10.49  14.73 19.11 27.94 46.73
                              hypre_GMRESSetup                        1466  5.93e+04 1.218e+05 2.897e+06  1.37e+06 6.653e+06 7.334e+07 2.161e+08    4.349e+05 4.618e+08 1.091e+09 1.232e+10 1.025e+10 3.404e+10 1.827e+11 2.517e+11       0.1653   5.641  10.49  10.61  15.03 19.11 24.78 46.73
                              hypre_BoomerAMGBuildMultipass            200      6065      9048 8.244e+05 1.854e+05 2.128e+06 2.982e+07 8.821e+07    6.252e+05 5.016e+07 1.259e+08 2.496e+09 9.876e+08 8.065e+09 3.473e+10  5.85e+10       0.1653   5.871  10.28   10.7  15.03 19.49 24.78 46.73
                              hypre_PCGSolve                          1676 4.759e+04 8.332e+04 1.689e+06 2.997e+05 6.653e+06 1.773e+07 1.952e+07    1.118e+07 4.263e+08 8.415e+08  1.52e+10 3.624e+09 4.595e+10 1.864e+11 2.423e+11       0.2372   6.235  10.28  10.95  15.64 19.89 28.51 29.16
         minife C++   CPP_APP main                               4.216e+05 8.311e+05 8.479e+05 7.971e+05  8.65e+05  8.65e+05 8.825e+05 8.897e+05    8.596e+09 8.596e+09 8.596e+09  8.77e+09 8.737e+09 9.465e+09 9.465e+09 9.525e+09        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              driver                             2.499e+05 2.503e+05 2.554e+05 2.626e+05 2.605e+05 3.057e+05 3.057e+05 3.064e+05    7.296e+09 7.445e+09 7.749e+09 7.888e+09 8.065e+09 8.547e+09 8.547e+09 8.547e+09        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              cg_solve                           2.229e+05 2.265e+05 2.311e+05 2.381e+05 2.357e+05 2.822e+05 2.822e+05 2.835e+05    6.767e+09 7.011e+09 7.153e+09 7.182e+09 7.298e+09 7.673e+09 7.673e+09 7.673e+09        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              dot                                      142     186.8     198.4     214.6     214.9     320.6     361.5       405     5.16e+06 5.784e+06 6.141e+06 6.741e+06 6.787e+06 1.033e+07 1.188e+07 1.279e+07        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              operator()                               139     149.9     156.1     255.6     159.2     169.1     333.7 3.682e+04    4.373e+06 4.831e+06 4.929e+06 7.798e+06  5.13e+06 5.339e+06 1.165e+07 1.087e+09        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              dot_r2                                   148     194.5     206.5     235.5     232.8     347.3     391.6      2625     5.22e+06 6.141e+06 6.653e+06 7.559e+06 7.807e+06 1.119e+07 1.287e+07  1.46e+07        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              exchange_externals                        93     98.51     100.5     200.8     104.6     111.1     278.7  3.67e+04    2.961e+06 3.111e+06 3.174e+06 6.031e+06  3.37e+06 3.579e+06 9.347e+06 1.083e+09        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              daxpby                                    27     36.24     39.26     40.93     41.69     45.16     159.2       312    9.266e+05 1.168e+06  1.24e+06 1.307e+06 1.317e+06 1.426e+06  5.13e+06 8.295e+06        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              generate_matrix_structure          1.233e+04 1.246e+04 1.271e+04 1.408e+04 1.271e+04 1.896e+04 1.896e+04 1.896e+04    2.551e+08 2.746e+08 3.354e+08 4.125e+08 5.104e+08 6.221e+08 6.221e+08 6.221e+08        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
                              init_matrix                             9813 1.041e+04 1.083e+04 1.089e+04  1.15e+04 1.197e+04 1.197e+04 1.197e+04    1.074e+08 1.209e+08 2.117e+08 2.406e+08 3.287e+08 3.706e+08 3.706e+08 3.706e+08        2.617   2.975  3.096  3.501  3.422 5.061 5.061 5.061
```

Each app on each system shows its ten functions with the most time.
tuolumne has no laghos rows, since laghos didn't run there.
matrix has no value for `PAPI_L1_DCM`, since no counter set of matrix holds it.
The same function changes with the system: the median call of `Mult` in laghos takes 9 ms on matrix and 28 ms on tioga.
`wide` holds every counter, so pick others by name in the column list, such as `["PAPI_L1_DCM"]`.

## 6. Read the counters as rates

Counts per call grow with the length of the call, so divide them by the cycles to compare functions of different sizes.
Use the sums over all calls, so each rate counts every call once.

```python
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
print(rates.loc[top].round(3).to_string())
```

```text
                                                                   ipc  flops_per_cycle  l1_miss_per_kinst  br_miss_per_kinst
system   app    layer cat     name                                                                                           
matrix   amg    C     C_APP   main                               1.669            0.263                NaN              1.754
                              hypre_GMRESSolve                   1.703            0.343                NaN              0.745
                              hypre_ParCSRMatrixMatvecOutOfPlace 1.614            0.343                NaN              0.643
                              hypre_BoomerAMGSolve               1.639             0.33                NaN              0.915
                              hypre_BoomerAMGCycle               1.639             0.33                NaN              0.915
                              hypre_CSRMatrixMatvecOutOfPlace    1.556            0.351                NaN               0.64
                              hypre_ParCSRRelax                  1.587            0.342                NaN              0.478
                              hypre_GMRESSetup                   1.634            0.026                NaN              6.157
                              hypre_PCGSolve                     1.373            0.186                NaN              0.586
                              hypre_BoomerAMGCoarsenPMIS         1.672            0.006                NaN              4.686
         laghos C++   CPP_APP main                                2.47            0.001                NaN              0.239
                              Mult                               2.484            0.001                NaN              0.238
                              SolveVelocity                      2.484            0.001                NaN              0.238
                              SolveEnergy                        2.493            0.001                NaN               0.23
                              GetTimeStepEstimate                2.669                0                NaN              0.078
                              LagrangianHydroOperator            0.426                0                NaN              0.244
                              UpdateQuadratureData               1.162            0.001                NaN              0.288
                              Rho0DetJ0Vol                       0.375                0                NaN              0.294
                              InternalEnergy                     0.685                0                NaN                0.2
                              SetEssentialTrueDofs               0.681                0                NaN              0.271
         minife C++   CPP_APP driver                             0.454                0                NaN               0.07
                              cg_solve                           0.574                0                NaN              0.077
                              make_local_matrix                  0.337                0                NaN              0.056
                              dot                                0.582                0                NaN              0.077
                              generate_matrix_structure          0.469                0                NaN              0.078
                              operator()                         0.606                0                NaN              0.072
                              finish_exchange_externals          0.641                0                NaN              0.066
                              assemble_FE_data                    0.36                0                NaN              0.067
                              perform_element_loop_cuda           0.36                0                NaN              0.067
                              waxpby                             0.533                0                NaN              0.088
tioga    amg    C     C_APP   main                               2.106            0.069              7.855               0.48
                              hypre_GMRESSolve                   1.642            0.121              12.71              0.274
                              hypre_BoomerAMGSolve               1.997            0.107              9.464              0.216
                              hypre_BoomerAMGCycle               1.997            0.107              9.464              0.216
                              hypre_ParCSRMatrixMatvecOutOfPlace 1.868            0.117              10.92              0.186
                              hypre_ParCSRRelax                   1.79            0.119              11.52              0.151
                              hypre_CSRMatrixMatvecOutOfPlace    0.849            0.178              35.88              0.374
                              hypre_GMRESSetup                   2.521            0.005              5.749              1.092
                              hypre_NewCommPkgCreate_core        2.593                0              8.566              0.556
                              hypre_DataExchangeList             2.593                0              8.567              0.556
         laghos C++   CPP_APP main                               2.028            0.003              5.264              0.933
                              Mult                               2.036            0.003              5.177              0.921
                              SolveVelocity                       2.04            0.003              5.168               0.92
                              SolveEnergy                        1.949            0.002              5.322              0.942
                              UpdateQuadratureData               1.488            0.003              8.719              1.447
                              GetTimeStepEstimate                1.968            0.002              4.427              0.805
                              SetEssentialTrueDofs               1.449            0.003              8.374              1.377
                              ForceMultTranspose                 1.504            0.003              8.677              1.445
                              ForceMult                          1.456            0.003              8.314              1.392
                              LagrangianHydroOperator            1.033            0.001              2.433              0.413
         minife C++   CPP_APP main                               1.529            0.153              1.144              0.186
                              driver                             1.377            0.165              1.296               0.21
                              generate_matrix_structure          1.325            0.068              1.416              0.194
                              cg_solve                           0.848            0.008              1.592              0.311
                              init_matrix                        1.254            0.046              1.275              0.206
                              assemble_FE_data                   2.607            0.759              1.251                0.1
                              perform_element_loop               2.607            0.759              1.251                0.1
                              operator()                         0.831            0.007              1.554                0.3
                              make_local_matrix                  1.098            0.021              0.894              0.219
                              impose_dirichlet                   1.375             0.05              0.966              0.374
tuolumne amg    C     C_APP   main                                1.74            0.277              32.39              1.672
                              hypre_GMRESSolve                   1.877            0.461              43.29              0.815
                              hypre_ParCSRMatrixMatvecOutOfPlace 1.931            0.451              41.52              0.649
                              hypre_BoomerAMGSolve               1.959            0.429              39.25                0.9
                              hypre_BoomerAMGCycle               1.959            0.429              39.26                0.9
                              hypre_CSRMatrixMatvecOutOfPlace    1.802            0.489               48.2              0.692
                              hypre_ParCSRRelax                  1.884            0.453               42.7              0.487
                              hypre_GMRESSetup                   1.238            0.016              20.61              5.074
                              hypre_BoomerAMGBuildMultipass      0.727            0.007              25.88              3.665
                              hypre_PCGSolve                     2.272            0.165              12.94              0.291
         minife C++   CPP_APP main                               0.909            0.012              0.462              0.245
                              driver                             0.894             0.01              0.453              0.188
                              cg_solve                           0.887            0.008              0.456              0.192
                              dot                                0.883            0.002              0.632              0.199
                              operator()                         0.894            0.002               0.56              0.252
                              dot_r2                             0.884            0.002              0.596              0.201
                              exchange_externals                 0.897            0.002               0.55              0.272
                              daxpby                             0.882            0.002              0.609               0.19
                              generate_matrix_structure          0.996            0.036              0.383              0.129
                              init_matrix                        0.996            0.038              0.569               0.19
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
- Back to the start: [Count the runs in a genesis trace](01-count-the-runs.md)
