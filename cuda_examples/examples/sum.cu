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
	// 每个线程负责将一部分数据写入共享内存
	shared[tx] = nums[id];
	__syncthreads();
	// 相近的两个元素之间进行sum
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
	// 前半部分的数组和后半部分的数据进行sum操作
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

__global__ void sum_3(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	// 一个线程块负责2 * BLOCK_SIZE个数据的reduce操作
	shared[tx] = nums[id] + nums[id + blockDim.x];
	__syncthreads();
	// 前半部分的数组和后半部分的数据进行sum操作
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

__global__ void sum_4(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	// 一个线程块负责2 * BLOCK_SIZE个数据的reduce操作
	shared[tx] = nums[id] + nums[id + blockDim.x];
	__syncthreads();
	// 前半部分的数组和后半部分的数据进行sum操作
	for (int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tx < stride)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	// 最后一轮展开
	if (tx < 32)
	{
		shared[tx] += shared[tx + 32];
		shared[tx] += shared[tx + 16];
		shared[tx] += shared[tx + 8];
		shared[tx] += shared[tx + 4];
		shared[tx] += shared[tx + 2];
		shared[tx] += shared[tx + 1];
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}

/**
 * @brief 完全展开
 */
__global__ void sum_5(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	// 一个线程块负责2 * BLOCK_SIZE个数据的reduce操作
	shared[tx] = nums[id] + nums[id + blockDim.x];
	__syncthreads();
	int blockSize = blockDim.x;
	// 前半部分的数组和后半部分的数据进行sum操作
	// 循环完全展开
	if (blockSize >= 512 && tx < 256)
	{
		shared[tx] += shared[tx + 256];
	}
	__syncthreads();
	if (blockSize >= 256 && tx < 128)
	{
		shared[tx] += shared[tx + 128];
	}
	__syncthreads();
	if (blockSize >= 128 && tx < 64)
	{
		shared[tx] += shared[tx + 64];
	}
	__syncthreads();
	// 最后一轮展开
	if (tx < 32)
	{
		shared[tx] += shared[tx + 32];
		shared[tx] += shared[tx + 16];
		shared[tx] += shared[tx + 8];
		shared[tx] += shared[tx + 4];
		shared[tx] += shared[tx + 2];
		shared[tx] += shared[tx + 1];
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
	float *res3;
	float *res4;
	float *res5;
	cudaMallocManaged(&nums, sizeof(float) * n);
	cudaMallocManaged(&res1, sizeof(float));
	cudaMallocManaged(&res2, sizeof(float));
	cudaMallocManaged(&res3, sizeof(float));
	cudaMallocManaged(&res4, sizeof(float));
	cudaMallocManaged(&res5, sizeof(float));
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
	dim3 gridSize1((blockSize.x + n - 1) / blockSize.x / 2 + 1);
	sum_3<<<gridSize1, blockSize>>>(nums, res3, n);
	cudaDeviceSynchronize();
	double t4 = getCurrentTime();
	sum_4<<<gridSize1, blockSize>>>(nums, res4, n);
	cudaDeviceSynchronize();
	double t5 = getCurrentTime();
	sum_5<<<gridSize1, blockSize>>>(nums, res5, n);
	cudaDeviceSynchronize();
	double t6 = getCurrentTime();
	cout << "result1: " << *res1 << ", time: " << t2 - t1 << endl;
	cout << "result2: " << *res2 << ", time: " << t3 - t2 << endl;
	cout << "result3: " << *res3 << ", time: " << t4 - t3 << endl;
	cout << "result4: " << *res4 << ", time: " << t5 - t4 << endl;
	cout << "result5: " << *res4 << ", time: " << t6 - t5 << endl;
	return 0;
}