NVCC          := /usr/bin/nvcc

# internal flags
NVCCFLAGS   := -arch=sm_60
CCFLAGS     := -g -G

TARGETS   = histogram

$(TARGETS): $(TARGETS).cu
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -o $@ $<


.PHONY: clean
clean:
	rm $(TARGETS)