# dftracer-genesis-analysis

Tutorials that analyze the genesis sweep traces with the dftracer-utils Python API, and a check that the tutorials still give the output they show.

## Tutorials

Work through them in order, on a corona login node.

1. [Count the runs in a genesis trace](tutorials/01-count-the-runs.md), about 10 minutes.
2. [Find where amg spends its time](tutorials/02-where-amg-spends-its-time.md), about 15 minutes.
3. [Count instructions per cycle for each call path](tutorials/03-count-instructions-per-cycle.md), about 10 minutes.
4. [Read the full sweep in a Flux job](tutorials/04-read-the-full-sweep-in-a-flux-job.md), about 5 minutes plus a 6-minute job.
5. [Build the call tree of one amg run](tutorials/05-build-the-call-tree-of-one-run.md), about 15 minutes.
6. [Measure the flop rate as processes share a node](tutorials/06-flop-rate-as-processes-share-a-node.md), about 10 minutes.
7. [Measure how much of main is MPI](tutorials/07-mpi-share-of-main.md), about 10 minutes.
8. [Compare the three systems](tutorials/08-compare-the-systems.md), about 15 minutes.
9. [Check how repeatable the runs are](tutorials/09-spread-and-repeatability.md), about 10 minutes.
10. [Read merged distributions](tutorials/10-merged-distributions.md), about 10 minutes.
11. [Measure the scaling efficiency](tutorials/11-scaling-efficiency.md), about 10 minutes.
12. [Read the host counters of each run](tutorials/12-host-counters.md), about 10 minutes.
13. [Read the counters of each function by layer](tutorials/13-counters-of-each-function.md), about 10 minutes.

Tutorials 1 to 3 and 5 use a 162 MB sample of the sweep.
The full sweep is one 17 GB file, `/usr/workspace/genesis-wisdom/dftracer-traces/genesis-sweep/genesis.pfw.gz`, with 9,624 runs.
Tutorial 4 reads it in a batch job, and tutorials 6 to 13 read it in Python on a compute node, after tutorial 4 has built its index.

## Check the tutorials

`tools/run_tutorial.py` runs the Python blocks of a page in one session and compares each output with the text block after it.

```bash
uv run python tools/run_tutorial.py tutorials/*.md
```

Add `--record` to write the real output into the pages instead.
Run it on corona, where the sample is.
