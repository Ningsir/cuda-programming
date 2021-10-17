NVCC          := /usr/bin/nvcc
OUT_DIR = $(CURDIR)/bin

# internal flags
NVCCFLAGS   := -arch=sm_60
CCFLAGS     := -g -G

TARGETS   = histogram

$(TARGETS): $(TARGETS).cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -o $(OUT_DIR)/$@.out $<


.PHONY: clean
clean:
	rm $(OUT_DIR)