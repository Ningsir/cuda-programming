// Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include <iostream>

#include "common/utils.h"

#define BLOCK_SIZE 256

using namespace std;

__global__ void sum(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	shared[tx] = nums[id];
	__syncthreads();
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (tx % (stride * 2) == 0)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}

/**
 * @brief 优化：减少分支
 * 
 */
__global__ void sum_2(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	shared[tx] = nums[id];
	__syncthreads();
	for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
	{
		if (tx < stride)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}

int main()
{
	unsigned int n = 80000;
	float *nums;
	float *res1;
	float *res2;
	cudaMallocManaged(&nums, sizeof(float) * n);
	cudaMallocManaged(&res1, sizeof(float));
	cudaMallocManaged(&res2, sizeof(float));
	for (int i = 0; i < n; i++)
	{
		nums[i] = 1.0;
	}
	dim3 blockSize(BLOCK_SIZE);
	dim3 gridSize((blockSize.x + n - 1) / blockSize.x);
	double t1 = getCurrentTime();
	sum<<<gridSize, blockSize>>>(nums, res1, n);
	cudaDeviceSynchronize();
	double t2 = getCurrentTime();
	sum_2<<<gridSize, blockSize>>>(nums, res2, n);
	cudaDeviceSynchronize();
	double t3 = getCurrentTime();
	cout << "result1: " << *res1 << ", time: " << t2 - t1 << endl;
	cout << "result2: " << *res2 << ", time: " << t3 - t2 << endl;
	return 0;
}