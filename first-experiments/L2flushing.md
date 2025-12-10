the memset‑based flush makes the starting L2 state deterministic (cold). But NCU also:

- Replays kernels to collect metrics

- May add and remove extra code around the kernel (software patching for certain metrics)

- May change when in the GPU timeline the victim runs relative to the enemy, especially under concurrency and application replay modes

In the concurrent case, the enemy is constantly thrashing L2 in the background; where its lines are in the set/index space during the victim’s execution can shift slightly from run to run. Small timing shifts change which of the victim’s lines are already in L2 vs just evicted.

Application replay re‑launches the whole workload, and exact overlap of victim vs enemy can differ a bit between profiling runs, changing how harshly the victim gets hit.


why do we need to flush L2 between runs?

-L2 is a set-associative cache. To evict a line you must fill its same set with enough distinct lines to exceed the associativity (or otherwise displace that line). A cache flush or eviction kernel must therefore touch enough distinct addresses that map across all sets (or specifically the same set as the victim’s lines) to cause eviction.

It’s very easy to under-evict by using unlucky strides, poor alignment, or by not touching enough unique cache lines — or to accidentally create access patterns that the hardware treats specially (prefetch, coalescing, streaming stores that bypass L2).




resources : 

[stackoverflowq](https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu)
[nvdia kernel benchmarking library](https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh)
[How To Benchmark CUDA Kernels](https://guillesanbri.com/CUDA-Benchmarks/)
[Cold VS Hot Measurements](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/)