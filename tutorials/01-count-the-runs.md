# Count the runs in a genesis trace

You'll read a sample of the genesis sweep with the dftracer-utils Python API, in pandas style, and print how many runs each application has on each system.
It takes about 10 minutes of your time.

## Before you start

You work on a corona login node, in a copy of this repository.
You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which corona has.

```bash
uv --version
```

```text
uv 0.11.27 (x86_64-unknown-linux-gnu)
```

From the top of this repository, install the Python packages.

```bash
uv sync
```

```text
Resolved 7 packages in 2ms
Checked 5 packages in 2ms
```

The first time, uv downloads the packages, so it prints more lines and takes a minute.

The sample is the smallest part of the sweep, 162 MB, in the shared `genesis-wisdom` workspace.
Link it into `data/`, so the index that the first query writes lands in your folder and not next to the trace.

```bash
mkdir -p data && ln -sf /usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/sample-s0-part0.pfw.gz data/genesis-sample.pfw.gz
```

`ln` prints nothing.
Check that the link reaches the file.

```bash
ls -lL data/genesis-sample.pfw.gz
```

```text
-rw-r----- 1 sinurat1 genesis-wisdom 161541455 Oct 12 01:17 data/genesis-sample.pfw.gz
```

The size must be 161541455 bytes.
If `ls` says `Permission denied`, ask to be added to the `genesis-wisdom` group, which can read the file.

## 1. Start Python

Every step after this one runs in one Python session, so keep it open until the end.
Start it with the log level set to `warn`, which hides the library's startup line.

```bash
DFTRACER_UTILS_LOG_LEVEL=warn uv run python
```

```text
Python 3.12.14 (main, Aug 20 2026, 07:40:49) [GCC 8.5.0 20210514 (Red Hat 8.5.0-28)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Type the code of each step at the `>>>` prompt.

## 2. Read the runs

A genesis trace keeps one record for each run.
An [`Indexer`](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html) builds the trace's index and keeps those records as the `runs` table.

```python
from dftracer.utils import Indexer

with Indexer(files=["data/genesis-sample.pfw.gz"]) as ix:
    ix.ensure_indexed()
    runs = ix.rowset("runs")
print(runs.shape)
```

```text
(131, 10)
```

That's 131 runs, each with 10 columns.
The index lands next to the link, in `data/.dftindex`, and the next steps reuse it.

## 3. Look at a few runs

`runs` is a dftracer-utils `DataFrame`, which takes pandas spellings such as `df[["a", "b"]]` and `head()`, and prints as pandas does.

```python
print(runs[["run", "app", "system", "unique_input", "nodes", "ppn", "papi_set"]].head())
```

```text
                run  app  system     unique_input  nodes  ppn                                        papi_set
0  59cecf6a74e1a929  amg   tioga  p2_n128x128x128      4   16  PAPI_BR_CN+PAPI_BR_INS+PAPI_BR_MSP+PAPI_BR_UCN
1  3f3589dc2db6645e  amg   tioga  p2_n128x128x128      4   16                         PAPI_BR_NTK+PAPI_BR_TKN
2  c5bbad7b9db1071d  amg   tioga  p2_n128x128x128      4   16                        PAPI_BR_PRC+PAPI_VEC_INS
3  388ca136cfe37072  amg   tioga  p2_n128x128x128      4   16          PAPI_FAD_INS+PAPI_FDV_INS+PAPI_FSQ_INS
4  85d23375d53b6df6  amg   tioga  p2_n128x128x128      4   16            PAPI_FML_INS+PAPI_FP_INS+PAPI_FP_OPS
```

A run is one application, one system, one input, one number of nodes, one number of processes per node (`ppn`) and one set of hardware counters (`papi_set`).
The `run` column is its id, and every other record in the trace names its run by this id.

## 4. Count the runs per application and system

`groupby(...).size()` counts the rows of each group, as in pandas.

```python
print(runs.groupby(["app", "system"]).size())
```

```text
      app    system  size
0     amg    matrix    27
1     amg     tioga     8
2     amg  tuolumne     9
3  laghos    matrix    87
```

The sample has amg on three systems and laghos on `matrix` only.

## 5. List the columns

```python
print(runs.columns)
```

```text
['run', 'app', 'system', 'unique_input', 'nodes', 'ppn', 'papi_set', 'method', 'sketch_accuracy', 'leaf']
```

## 6. Count the other records

Apart from the run records, the trace holds `func` records, one for each call path of a run, and `counter` records, one for each call path and counter.
A [`TraceViewer`](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html) reads them, and its `group_by` counts them without loading the records into memory.

```python
from dftracer.utils import TraceViewer

tv = TraceViewer("data/genesis-sample.pfw.gz")
print(tv.group_by("gtype").agg("count").sort_by("gtype").collect())
```

```text
     gtype    count
0  counter  4905256
1     func    23075
```

There are about 200 counter records for each call path, because every path has its own record for each counter.

## 7. Count the amg runs for each counter set

Count the amg runs on `matrix` for each `papi_set`.

```python
amg = runs[(runs["app"] == "amg") & (runs["system"] == "matrix")]
print(amg.groupby("papi_set").size())
```

```text
                 papi_set  size
0  papi_set1_branch_cycle    12
1          papi_set2_flop    15
```

The two sets don't match.
Not every configuration ran with both counter sets, so compare runs of one set with runs of the same set.

## What you built

You read the runs of a genesis trace into a table and counted them and the other records with the dftracer-utils Python API.
Keep the Python session open for [Find where amg spends its time](02-where-amg-spends-its-time.md).

- How the genesis records are made: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
- The Python classes: [Python API Reference](https://dftracer.readthedocs.io/projects/utils/en/latest/api/index.html)
