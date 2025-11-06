here i will leave remarks and notes about my state of progress in the quest of building contention kernels

# L2 contention kernels

-the goal is that our aggressor repeatedly access memory locations that map to the ensemble of cache sets
-Caches are divided into "sets" to organize data. When multiple threads repeatedly access different memory addresses that all map to the same cache set, they constantly evict each other's data, causing a large number of cache misses.
-i have to not rely too much on hardware because iam working on GeForce but my target platform is Orin or rely but find a general
patterns so it's not platform specific
-i have to know the scenarios i must simulate , the L2 cache size of the gpu i'm working on
-for that we have to allocate a large working set: the total size of the arrays accessed by the contending kernels must significantly exceed the size of the L2 cache on our GPU. we query the L2 cache size using the CUDA device properties API (cudaGetDeviceProperties).
-do we need MPS to corun the victim and aggressor (i suppose to be ran in the same context ? but MPS has overhead)

so to we use a validated L2-thrashing enemy design with pointer-chasing and controlled address mapping, co-run it with a cache-reuse victim under MPS, and vary “coverage × intensity” to realize 0/50/100% L2 pressure; 
we verify with Nsight metrics and per-instruction stalls before reporting slowdown and p99 timing shifts.

Choose a kernel with strong L2 reuse: tiled convolution, stencil, or blocked matmul with reuses across tiles; confirm L2 hit rate is high when run alone and we should confirm that the observed metrics are

scenarios:

0% : no stress from enemy, victim behaves alone
25% : enemy takes over 25% of the cache 
50%.. 100%

Verification metrics and signals

-Nsight Compute: L2 hit rate, L2 read/write transactions, sector queries per request, local-memory activity (to ensure no victim spill confound), DRAM bandwidth, MSHR saturation proxies.​

-CUPTI PC sampling: attribute stalls near victim load PC to long_scoreboard/lg_throttle increases when enemies are active.​

-Timing: record median and p90/p99 victim runtime with CUDA events; report “sensitivity” = (victim time with enemy at p90) / (victim time with occupation at median).

Deliverables

Enemies: three binary variants (0/50/100) with a JSON config to switch between duty-cycle, set-coverage, and SM-fraction modes.​

Reports: per-level plots of L2 hit rate, DRAM traffic, stall breakdown, and victim p50/p90/p99 times; include sensitivity vs. level curves.​

Repro scripts: MPS setup, launch ordering, and Nsight CLI invocations; occupation baseline runs for fair comparisons

Distinguish L2 vs. DRAM: when isolating L2 effects, keep enemy array at L2 size; when exploring “beyond L2,” increase array to 12–16 MB to intentionally push MSHR/DRAM.

for my GPU:
--- Device 0 ---
  Name: NVIDIA GeForce GTX 1650 Ti
  Compute Capability: 7.5
  Total Global Memory: 4095 MB
  Multiprocessor Count: 16
  Max Threads Per Block: 1024
  L2 Cache Size: 1048576 bytes (1 MB)



keep in mind: cudaLimitPersistingL2CacheSize

GeForce 
Render Config
Shading Units 1024
TMUs 64
ROPs 32
SM Count 16
L1 Cache 64 KB (per SM)
L2 Cache 1024 KB
Memory Size 4 GB Memory Type GDDR6 Memory Bus 128 bit Bandwidth 192.0 GB/s

Control of occupation level

To emulate 0%, 50%, 100% L2 usage:

Compute bytes_to_touch = l2CacheSize * frac (frac ∈ {0, 0.5, 1.0}).

Create buf of that size, run kernel for long duration.

For “100%” plus oversaturation, use 2–4× l2CacheSize (provokes constant evictions).

These fractional levels approximate occupancy; on Windows you can’t isolate sets anyway.

Validation (proof that L2 is fully active)

Profile one run of l2_saturate_write alone:

In Nsight Compute, collect:

l2_tex_read_hit_rate

l2_tex_write_hit_rate

dram__throughput.avg.pct_of_peak_sustained_elapsed

l2_subp0_write_sectors

sm__warps_active.avg.pct_of_peak_sustained_active

Expected pattern:

Hit rate ≈ 0–5 % (streaming through L2)

DRAM throughput ≈ 70–90 % of peak

Warps active ≈ 90 % +
That empirically proves full L2 churn


# L1 / shared memory contention kernels

keep in mind :  cudaDeviceGetCacheConfig 



 cudaDeviceProp::

Thread/warp scaling:	warpSize, maxThreadsPerBlock, maxThreadsPerMultiProcessor, multiProcessorCount, maxBlocksPerMultiProcessor
Register occupancy tuning:	regsPerBlock, regsPerMultiprocessor
Shared-memory usage tuning:	sharedMemPerBlock, sharedMemPerBlockOptin, sharedMemPerMultiprocessor
we may use these to size thread blocks, warps, and SM usage so you can saturate compute units or hide latency when stressing memory.
Example: multiProcessorCount × maxBlocksPerMultiProcessor defines the maximum number of resident blocks; your enemy launch grid should approach this.

Interference type	Key variables
Compute (ALU, scheduler)	multiProcessorCount, maxThreadsPerMultiProcessor, warpSize, regsPerBlock, major/minor
L1/shared cache	sharedMemPerMultiprocessor, localL1CacheSupported, globalL1CacheSupported
L2 / global memory	l2CacheSize, memoryBusWidth, totalGlobalMem
DRAM / MSHR bandwidth	asyncEngineCount, totalGlobalMem, memoryBusWidth
Cross-channel concurrency	concurrentKernels, computePreemptionSupported
Implementation usage map

Launch scaling: derive grid = multiProcessorCount × (maxBlocksPerMultiProcessor); threads = multiple of warpSize.

Memory region sizing: choose array sizes relative to l2CacheSize for cache stress, or large multiples (×3–×6) for DRAM stress.

Warp design: use warpSize to index pointer-chase loops precisely (per-warp vs per-thread).

Stream and concurrency logic: only attempt concurrent kernels if concurrentKernels==1.

Compute kernel variant: select PTX ops according to major/minor (e.g., Ampere supports IMMA, Turing supports HMMA).


Target	Variables	Use
L2	l2CacheSize, persistingL2CacheMaxSize	For coverage and stride design; persistingL2CacheMaxSize limits persistent lines reserved by driver.
L1/shared	localL1CacheSupported, globalL1CacheSupported, sharedMemPerMultiprocessor	Detect if SM-local cache is unified or split with shared memory; critical for SM-local contention kernels.
Global memory	totalGlobalMem, memoryBusWidth, memPitch	Determines total DRAM capacity and alignment; helps compute stride to provoke DRAM bank conflicts.
3. Concurrency and scheduling — control how many kernels can coexist
Purpose	Variables
concurrentKernels — must be 1 to allow concurrent launches; false means serialized.	
asyncEngineCount — number of copy engines; needed when you add DMA or copy-based stressors.	
mpsEnabled — true only on Linux; if 0, you must use same-context streams to create concurrency.	
computePreemptionSupported — if 1, hardware may time-slice SMs; you must account for interference noise.	
4. Compute resource structure — used when designing ALU-heavy or mixed stressors
Target	Variables
major, minor (compute capability) — choose correct PTX/SASS ops (e.g., FP16, INT32).	
regsPerMultiprocessor, warpSize — used to saturate pipelines.