NVCC          := /usr/bin/nvcc
WB_DIR := ../third_party/libwb
OUT_DIR = $(CURDIR)/bin

# building wb library
wb:
	@echo "prepare wb library ..."
	@cd "${WB_DIR}"; make all; sudo cp ./lib/libwb.so /usr/local/lib/;
	@echo "wb built done"

# internal flags
NVCCFLAGS   := -arch=sm_60
CCFLAGS     := -std=c++11 -g -G

TARGETS   = mp0
$(TARGETS): src/$(TARGETS).cu
	mkdir -p $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I ../third_party/libwb/ -lwb -o $(OUT_DIR)/$@.out $<

.PHONY: clean
clean:
	rm -rf bin/