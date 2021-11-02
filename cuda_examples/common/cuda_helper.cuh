#ifndef CUDA_EXAMPLES_COMMON_CUDA_HELPER_CUH_
#define CUDA_EXAMPLES_COMMON_CUDA_HELPER_CUH_

#include <cuda.h>

#define checkCudaErrors(ans)                  \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#endif // CUDA_EXAMPLES_COMMON_CUDA_HELPER_CUH_
