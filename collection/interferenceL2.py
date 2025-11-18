# in this file we'll launch 2 kernels : the enemy and the victim
#they should be launched concurrently through the same context by the use of NVIDIA MPS


# for now we dont have MPS here because i dont have linux yet i have windows so this is just a placeholder for now
# for now the exe that ill input just launch the victim alone AND the victim+enemy (they are in the same .cuda
# just in 2 different streams) and collect the metrics
#we'll collect the timing metrics using nsys because they are more accurate and less overhead
# and the lts_sector_lookup_miss using ncu
# the program contain a lot of params (n) so here for each param  ; we'll fix a combination (n-1) and loop over 1 to generate the data
#and we'll do it over all the params one by one to generate the data and append them into the csv
# the params are independent from the victim as much as we can