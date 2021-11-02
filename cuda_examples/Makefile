NVCC	:= $(shell which nvcc)
OUT_DIR = $(CURDIR)/bin

# internal flags
NVCCFLAGS   := -arch=sm_60
CCFLAGS     := -g -G

TARGETS   = histogram
SOURCES = $(shell find -name "*.cu")
INCLUDES = $(CURDIR)

$(TARGETS): $(TARGETS).cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<

matrixMul: examples/matrixMul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
convolution: examples/convolution.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
sum: examples/sum.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
sparseMatrixMul: examples/sparseMatrixMul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
triangleCount: examples/triangleCount.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<

bfs: examples/bfs.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
kernel.o: src/kernel.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@ -c  $<
sparse.o: examples/sparseMatrixMul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@ -c  $<
sparse: kernel.o sparse.o
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@ bin/kernel.o bin/sparse.o

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)