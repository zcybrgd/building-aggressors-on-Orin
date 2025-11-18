# so we have benchmarks that we wanna run on different types of interference; there are compute bound benchmarks that
#we want to put under compute contention, and there are memory bound benchmarks that we want to put under memory contention.

#ideal test case is we have NVIDIA MPS and 3 contention kernels: L2 cache Contention Kernel, L1 cache CK; compute CK
# there are 3 types of benchmarks those who we run against only one of those, those we run against 2 etc.. 

#lets not forget that the goal is to construct our transformer dataset
#the goal is at the end to have a csv that has : benchmark name, contention type (it can be none(baseline), l1,l2,compute),
# the combination of parameters grid size victim working set size etc.,the metrics for example if L2 contention 
# then wel'll have execution time + lts_sector lookup miss (execution time is always present)

# so this script for example for L2 benchmarks will launch all the benchmarks under L2 contention and collect the metrics
# i ll write that script that is dedicated to L2 in this file : interferenceL2.py it will take as an output one .exe of the victim



# with the csv we can do plots and analysis later on.