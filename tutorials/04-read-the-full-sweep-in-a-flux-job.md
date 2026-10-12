# Read the full sweep in a Flux job

You'll run the amg analysis of [Find where amg spends its time](02-where-amg-spends-its-time.md) over the whole genesis sweep, 9,624 runs in one 17 GB file, as a [Flux](https://flux-framework.readthedocs.io/en/latest/quickstart.html) batch job.
It takes about 5 minutes of your time, plus about 6 minutes for the first run of the job, nearly all of it to build the index.

## Before you start

Work on a corona login node, in the top folder of this repository, with the Python environment from [Count the runs](01-count-the-runs.md).
Check that you can read the full sweep, which the `genesis-wisdom` group can read.

```bash
ls -l /usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz
```

```text
-rw-r----- 1 sinurat1 genesis-wisdom 16965172015 Oct 12 01:06 /usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz
```

## 1. Read the job script

`jobs/full-sweep.sh` runs `scripts/full_sweep.py` on one `pdebug` node.

```bash
cat jobs/full-sweep.sh
```

```text
#!/bin/bash -e
# Count the runs of the full genesis sweep and print how amg's main scales on matrix.

#FLUX: --job-name=full-sweep
#FLUX: -N 1
#FLUX: -t 30m
#FLUX: -q pdebug
#FLUX: --output=full-sweep-{{id}}.out

# ---- 1. Settings: change these ----
TRACE=/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz
INDEX="$PWD/data/full-index"
AMG_INPUT=p1_n64x64x32
export DFTRACER_UTILS_LOG_LEVEL=warn

# ---- 2. Record the run ----
echo "Job $(flux getattr jobid) on $(hostname) at $(date)"

# ---- 3. Run ----
mkdir -p "$INDEX"
# The environment from uv sync; the job does not rebuild it.
uv run --no-sync python scripts/full_sweep.py "$TRACE" "$INDEX" "$AMG_INPUT"
```

The index goes to `data/full-index` in this repository, and later runs reuse it.
Run Python in a job through `uv run --no-sync`.
On corona, a plain `python3` in our job silently imported an old `dftracer-utils` 0.0.12 from another install.

## 2. Submit it and wait

[`flux job status`](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/man1/flux-job.html) waits until the job ends and exits with its exit code.

```bash
jobid=$(flux batch jobs/full-sweep.sh)
time flux job status "$jobid"; echo "exit $?"
```

```text
real    5m52.688s
user    0m0.054s
sys     0m0.016s
exit 0
```

## 3. Read the result

```bash
cat "full-sweep-$jobid.out"
```

```text
Job f67Kyv3RsJpf on corona186 at Mon Oct 12 01:08:36 PDT 2026
index ready in 344 s
      app    system  size
0     amg    matrix   612
1     amg     tioga   944
2     amg  tuolumne  2448
3  laghos    matrix  3235
4  laghos     tioga   823
5  minife    matrix   882
6  minife     tioga   672
7  minife  tuolumne     8
    nodes  ppn     p50_s
0       1    1  0.756291
1       1    2  0.847118
2       1    4  0.992754
3       1    8  1.264462
4       1   16  1.795116
5       1   32  2.902247
6       1   64  3.080486
7       2    1  0.891407
8       2    2  0.946531
9       2    4  1.105766
10      2    8  1.303508
11      2   16  1.906123
12      2   32  3.049681
13      2   64  3.473253
14      4    1  0.965653
15      4    2  1.046082
16      4    4  1.156246
17      4    8  1.412077
18      4   16  2.149157
19      4   32  3.651884
20      8    1  1.111214
21      8    2  1.156103
22      8    4  1.358335
23      8    8  1.624282
24      8   16  3.044887
amg scaling in 1 s
```

Your job ID, node and date will differ.
The first table counts the runs of each application on each system, 9,624 in all.
The second is the median time of amg's `main`, in seconds, for each setting on matrix.
The sample gave 1.81 s at 1 node with 16 processes, and the full sweep gives 1.80 s there.
It also holds the settings the sample didn't, up to 64 processes on one node.
The amg read took 1 s, because the index lets it read only the few chunks that hold `func` records.

## 4. Compare a smaller input

Set `AMG_INPUT=p1_n32x32x16` in `jobs/full-sweep.sh`, the input that is 8 times smaller, and submit the job again.

```bash
jobid=$(flux batch jobs/full-sweep.sh)
flux job status "$jobid"; cat "full-sweep-$jobid.out"
```

```text
Job f67L2j9AchwV on corona183 at Mon Oct 12 01:14:46 PDT 2026
index ready in 0 s
...
    nodes  ppn     p50_s
0       1    1  0.071310
1       1    2  0.089416
2       1    4  0.106838
3       1    8  0.151820
4       1   16  0.268798
5       1   32  0.465694
6       1   64  0.598237
...
amg scaling in 1 s
```

The index from the first run is reused, so the job ends in seconds.
At 1 node with 16 processes `main` takes 0.27 s instead of 1.80 s, more than 6 times less for 8 times less work.
Set `AMG_INPUT` back to `p1_n64x64x32` when you're done.

## What you built

You ran the amg analysis over the whole sweep on a compute node, and reran it in seconds on the index it left.

- The Flux commands: [flux-batch](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/man1/flux-batch.html), [flux-jobs](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/man1/flux-jobs.html) and [flux-job](https://flux-framework.readthedocs.io/projects/flux-core/en/latest/man1/flux-job.html)
- How the index lets a query skip chunks: [Build per-function distributions from genesis traces](https://dftracer.readthedocs.io/projects/utils/en/latest/guides/tools/genesis-gen-dist.html)
