L2 CACHE CONTENTION - CLEAN INDIVIDUAL PLOTS
================================================================================

VISUALIZATION APPROACH:
--------------------------------------------------------------------------------
Each plot shows all experiments with a SPECIFIC parameter value:
  - X-axis: Each experiment with this parameter value
  - Y-axis: The metric (execution time, L2 misses, or slowdown)
  - TWO bars per experiment: Alone (green/blue) vs Concurrent (red/orange)
  - Clear, readable, publication-ready

FILE NAMING:
--------------------------------------------------------------------------------
01_exec_time_PARAM_NN_VALUE.png  - Execution time plots
02_l2_misses_PARAM_NN_VALUE.png  - L2 misses plots
03_slowdown_PARAM_NN_VALUE.png   - Slowdown factor plots
(PARAM = parameter name, NN = plot number, VALUE = parameter value)

DATA SUMMARY:
--------------------------------------------------------------------------------
Total experiments: 60
Slowdown: 1.01x to 19.36x
Average slowdown: 4.83x
L2 miss increase: -77739 to 1049364
