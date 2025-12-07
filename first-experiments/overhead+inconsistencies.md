--replay-mode application runs ENTIRE program from start:

Normal run:        [Launch → Execute → Exit]
Application replay: [Launch → NCU intercepts → Inject counters → Execute → Read counters → Exit]

Overhead per kernel: ~10-50ms (CUDA API interception + counter setup)
