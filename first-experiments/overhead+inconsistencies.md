--replay-mode application runs ENTIRE program from start:

Normal run:        [Launch → Execute → Exit]
Application replay: [Launch → NCU intercepts → Inject counters → Execute → Read counters → Exit]

Overhead per kernel: ~10-50ms (CUDA API interception + counter setup)


# Important Considerations


1-it's true that we can measure L2 counters while an aggressor runs concurrently, but NCU may replay or serialize the workload if it needs multiple hardware passes (notably while we request lots of metrics). That can change concurrency, it's still undocumented.
2-we can't pin kernels to specific SMs to avoid interference from other kernels running on the same GPU 
3-there is no mechanism to know if the aggressor is still running concurrently while we measure the victim with NCU, so the results may be inconsistent, but the rising has shown that in practice the interference is present most of the time.
4-i shall use NCU only to collect a small set of metrics that can be collected in a single pass.
5-i shall make results repeatable//reproducible: flush L2 explicitly between runs,fix clocks/power to avoid thermal throttling so frequency and clocks don’t change between runs, do warm-up runs, collect many trials and report statistics (mean ± std / CI).
6-if the metrics consisted to be noisy we should run each configuration (baseline and contention) for N independent trials (say N ≥ 15), compute mean, variance.
7-log L2 counters and other counters together and compute derived measures (miss rate = misses / lookups)
8-in addition to L2 metrics, SM utilization and memory stall counters may be relevant too to correlate misses with performance loss
9-NCU's replay mechanism doesn't preserve the concurrent cache pressure
10- to interesting NCU commands

--cache-control arg (=all)            Control the behavior of the GPU caches during profiling. Allowed values:
                                          all
                                          none
--clock-control arg (=base)           Control the behavior of the GPU clocks during profiling. Allowed values:
                                          base
                                          (Lock GPU clocks to base)
                                          none
                                          (Don't lock clocks)
                                          reset
                                          (Reset GPU clocks and exit)

11-






