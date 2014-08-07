# this build depends on my own personal installation of CUDA

include cudaMagic.mk

# Include the helper functions from the samples directory
INCLUDES  := -I/usr/local/cuda/samples/common/inc

mergesort: src/mergesort.cu
	$(NVCC) $(INCLUDES) $(ALL_CFLAGS) $(GENCODE_FLAGS) $< -o $@

build:
	mkdir build

clean:
	rm -rf mergesort

