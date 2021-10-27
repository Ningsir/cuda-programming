NVCC := /usr/bin/nvcc
OUT_DIR = $(CURDIR)/bin

matrix:
	mkdir -p $(OUT_DIR)
	$(NVCC) -I ./ -o $(OUT_DIR)/matrix main.cpp matrix.cu

clean:
	rm -rf $(OUT_DIR)/