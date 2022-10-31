CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60


all: converter

converter: converter.cpp
	$(CC) converter.cpp -o converter $(CFLAGS) -I ../include/
			
clean:
	rm -f converter
