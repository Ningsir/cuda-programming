CXX := g++
OUT_DIR = $(CURDIR)/bin

# internal flags
CCFLAGS     := -g
INCLUDES = $(CURDIR)
test: test.cpp
	mkdir -p $(OUT_DIR)
	$(CXX) $(CCFLAGS) -I $(INCLUDES) -o $(OUT_DIR)/$@.out $<

clean:
	rm -rf $(OUT_DIR)