

the memset‑based flush makes the starting L2 state deterministic (cold). But NCU also:

- Replays kernels to collect metrics

- May add and remove extra code around the kernel (software patching for certain metrics)

- May change when in the GPU timeline the victim runs relative to the enemy, especially under concurrency and application replay modes

In the concurrent case, the enemy is constantly thrashing L2 in the background; where its lines are in the set/index space during the victim’s execution can shift slightly from run to run. Small timing shifts change which of the victim’s lines are already in L2 vs just evicted.

Application replay re‑launches the whole workload, and exact overlap of victim vs enemy can differ a bit between profiling runs, changing how harshly the victim gets hit.


resources : 

[stackoverflowq](https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu)
[nvdia kernel benchmarking library](https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh)
[How To Benchmark CUDA Kernels](https://guillesanbri.com/CUDA-Benchmarks/)
