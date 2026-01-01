# The Metrics Guide
i currently have access to 1,654 performance metrics on my NVIDIA GeForce GTX 1650 Ti, which is a Turing-based consumer GPU. These metrics are organized by hardware component prefixes and tell us everything about how the GPU executes code.

Metric Distribution by Hardware Component
Your metrics break down as follows:

Component	Prefix	Count	Relevance	Purpose
L2 Cache (Memory)	lts	766	⭐⭐⭐ Critical	Cache misses, hits, contention—the core of your cache analysis
SM Scheduler & Pipeline	smsp	322	⭐⭐⭐ Critical	Instruction execution, stalls, pipeline activity
L1 Cache (Per-SM)	l1tex	306	⭐⭐⭐ Critical	Cache conflicts, bank conflicts, hit rates
SM Management	sm	160	⭐⭐ High	Warp activity, overall SM execution
Infrastructure	tpc, gpu	37	⭐ Medium	Register usage, timing, GPU-wide cycles
Memory Subsystem	dram	12	⭐⭐ High	DRAM bandwidth, memory utilization
System & Debug	sys, idc, pcie	19	⭐ Low	System clocks, instruction divergence tracking
Graphics Only	fbpa, fe, gpc, gr	29	✗ None	Frame buffer, frontend, graphics pipeline—NOT for compute
Understanding Metric Types
1. Counters (~1000+ metrics)
Raw event counts that accumulate during kernel execution:

text
dram__bytes_read = 100,000,000        ← 100 MB read from DRAM
lts__t_requests = 50,000,000          ← 50M requests to L2 cache
l1tex__data_bank_conflicts = 1,000    ← 1K bank conflicts in shared memory
2. Throughput Metrics (~50 metrics)
Percentage of theoretical peak utilization:

text
dram__throughput = 85%                ← Using 85% of peak 192 GB/s
l1tex__throughput = 45%               ← L1 cache not fully utilized
gpu__compute_memory_throughput = 60%  ← Overall memory pipeline at 60%
3. Ratio Metrics (~400 metrics)
Derived values (hits/misses, averages, rates):

text
lts__t_request_hit_rate = 0.75        ← 75% of L2 requests hit
smsp__average_warp_latency = 50       ← Avg 50 cycles per warp resident
l1tex__t_sector_hit_rate = 0.40       ← 40% L1 cache hit rate
Why Some Metrics Report N/A
✗ Graphics-Only Metrics (29 metrics)
These hardware components are exclusively for rendering graphics, not compute:

Prefix	Purpose	Why N/A
fbpa__*	Frame Buffer Path (connects GPU to monitor)	Your kernel doesn't output to display
fe__*	Frontend (graphics command fetch)	Compute doesn't use graphics pipeline
gpc__*	Graphics Processing Cluster (rasterization)	No triangle rendering in compute
gr__*	Graphics Rasterizer (triangle setup)	Compute kernels don't rasterize
These metrics returning N/A is completely normal and expected. You should skip them entirely in your profiling.

✓ Valid Compute Metrics (~1,600 metrics)
All other metrics can produce valid data for your compute kernels. If they're returning N/A, check:

Does the kernel use that feature?

Texture metrics (l1tex__texin_) → Only if you use tex2D() or surface reads

Atomic metrics (lts__t_requests_op_atom) → Only if you use atomicAdd(), etc.

Is NCU profiling overhead too high?

Try profiling fewer metrics at once

Split into multiple runs

Metric requires special scope?

Some metrics need .avg, .sum, or .max suffix

Your script already handles this retry logic


Top 10 Metrics You MUST Track
For your PFE project analyzing cache contention with aggressor kernels:

1. lts__t_requests (Counter)
Total requests to L2 cache from all SMs. This is your baseline metric—tells you total memory traffic.

2. lts__t_requests_lookup_miss (Counter)
L2 cache misses that must go to DRAM. High misses = cache thrashing = potential contention.

Contention Indicator:

text
Hit Rate = Hits / (Hits + Misses)
< 30% hit rate = cache is not working → contention likely
3. lts__t_request_hit_rate (Ratio, 0.0-1.0)
Direct hit rate calculation. Multiply by 100 for percentage.

4. l1tex__data_bank_conflicts (Counter)
Shared memory bank conflicts = SERIALIZATION. Each thread accessing different bank = parallel. Same bank = waits for previous thread. High value = threads fighting for same bank.

5. l1tex__t_set_conflicts (Counter)
L1 cache set conflicts. Similar to bank conflicts but for L1 cache. Multiple threads mapping to same cache line = extra cycles.

6. dram__throughput (Throughput, %)
Percentage of peak 192 GB/s being used.

>80% = severely memory-bound

50-80% = memory is bottleneck

<30% = memory not saturated, other bottleneck

7. sm__warps_active (Counter)
Cumulative warps in flight. Tells you execution utilization.

8. smsp__average_warp_latency_issue_stalled_long_scoreboard (Ratio)
Warp stall cycles waiting for L1TEX (memory operations). High value = memory dependency stalls.

9. sm__cycles_elapsed (Counter)
Total GPU cycles kernel ran. Multiply by (1.5 GHz) to get seconds of GPU time.

10. dram__bytes_read + dram__bytes_write (Counters)
Actual DRAM traffic in bytes. Calculate real bandwidth:

text
Real_BW_GB/s = (bytes_read + bytes_write) / kernel_time_seconds
             = (bytes_read + bytes_write) / (cycles_elapsed / 1.5e9)
             