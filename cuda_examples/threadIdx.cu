#include <iostream>
using namespace std;

#define ARRAY_SIZE 128

__global__ void computeId(unsigned int *block, unsigned int *thread, unsigned int *warp)
{
	unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
	warp[thread_idx] = threadIdx.x / warpSize;
}

int main()
{
	int num_blocks = 2;
	int num_threads = 64;
	unsigned int *block;
	unsigned int *thread;
	unsigned int *warp;
	cudaMallocManaged((void **)&block, ARRAY_SIZE * sizeof(unsigned int));
	cudaMallocManaged((void **)&thread, ARRAY_SIZE * sizeof(unsigned int));
	cudaMallocManaged((void **)&warp, ARRAY_SIZE * sizeof(unsigned int));
	computeId<<<num_blocks, num_threads>>>(block, thread, warp);
	cudaDeviceSynchronize();
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		cout << "block: " << block[i] << ", thread: " << thread[i] << ", warp: " << warp[i] << endl;
	}
	return 0;
}