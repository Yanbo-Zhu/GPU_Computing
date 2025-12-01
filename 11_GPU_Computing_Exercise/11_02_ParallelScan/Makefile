build: slow_scan.cu fast_scan.cu
	nvcc -Werror all-warnings -O2 -o slow_and_fast_scan slow_scan.cu fast_scan.cu -lcurand -lcudart
