nvcc -arch=sm_87 -O3 -o verify_isolation verify_isolation.cu -lcuda


./verify_isolation


./verify_isolation 20000000

./verify_isolation 10000000 96 32