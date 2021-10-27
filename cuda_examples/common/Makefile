NVCC	:= $(shell which nvcc)
OUT_DIR = $(CURDIR)/bin

# internal flags
CCFLAGS     := -g
INCLUDES = $(CURDIR)
test: test.cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<

clean:
	rm -rf $(OUT_DIR)