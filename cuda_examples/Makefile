NVCC          := /usr/bin/nvcc
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

matrixMul: matrixMul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
convolution: convolution.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
sum: sum.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
sparseMatrixMul: sparseMatrixMul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<
.PHONY: clean
clean:
	rm -rf $(OUT_DIR)