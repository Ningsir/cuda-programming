CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60


all: sssp

sssp: src/sssp.cu
	$(NC) src/sssp.cu -o sssp $(CFLAGS) -I include/
			
clean:
	rm -f sssp
