#include <stdio.h>
#include <iostream>
#include <cassert>

#include "common/utils.h"

using namespace std;

#define MASK_SIZE 3
#define BLOCK_SIZE 256

__device__ __constant__ float Const_M[MASK_SIZE];

__global__ void convolution_1D(float *N, float *M, float *P, unsigned int N_Size, unsigned int Mask_Size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;
	int left = id - Mask_Size / 2;
	for (int i = 0; i < Mask_Size; i++)
	{
		if (left + i < N_Size && left + i >= 0)
			sum += M[i] * N[left + i];
	}
	P[id] = sum;
}

__global__ void convolution_1D_with_constant_memory(float *N, float *P, unsigned int N_Size, unsigned int Mask_Size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;
	int left = id - Mask_Size / 2;
	for (int i = 0; i < Mask_Size; i++)
	{
		if (left + i < N_Size && left + i >= 0)
			sum += Const_M[i] * N[left + i];
	}
	P[id] = sum;
}

/**
 * @brief 使用共享内存减少全局内存的访问，每个线程读取一个数据到共享内存。
 * 
 */
__global__ void convolution_1D_with_shared_memory(float *N, float *P, unsigned int N_Size, unsigned int Mask_Size)
{
	__shared__ float shared[BLOCK_SIZE + MASK_SIZE - 1];
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int n = Mask_Size / 2;
	int tx = threadIdx.x;
	// 前n个线程处理边界
	if (tx < n)
	{
		shared[tx] = id - n >= 0 ? N[id - n] : 0;
		// printf("thread id：%d, data: %f\n", threadIdx.x, shared[tx]);
	}
	// int tx = threadIdx.x;
	shared[tx + n] = N[id];
	// 后n个线程处理边界
	if (tx > BLOCK_SIZE - n - 1)
	{
		shared[tx + n + n] = id + n < N_Size ? N[id + n] : 0;
		// printf("thread id：%d, data: %f\n", threadIdx.x, shared[tx + n + n]);
	}
	__syncthreads();
	// printf("thread id：%d, data: %f\n", threadIdx.x, shared[tx + n]);
	float sum = 0.0;
	for (int i = 0; i < Mask_Size; i++)
	{
		sum += Const_M[i] * shared[threadIdx.x + i];
	}
	P[id] = sum;
	// printf("thread id：%d, sum: %f\n", threadIdx.x, sum);
}
int main()
{
	int N_Size = 80000;
	float *N, *P1, *P2, *P3;
	cudaMallocManaged(&N, sizeof(float) * N_Size);
	cudaMallocManaged(&P1, sizeof(float) * N_Size);
	cudaMallocManaged(&P2, sizeof(float) * N_Size);
	cudaMallocManaged(&P3, sizeof(float) * N_Size);
	for (int i = 0; i < N_Size; i++)
	{
		N[i] = 1;
	}
	float *M0 = (float *)malloc(MASK_SIZE * sizeof(float));
	for (int i = 0; i < MASK_SIZE; i++)
	{
		M0[i] = 1.0;
	}
	cudaMemcpyToSymbol(Const_M, M0, sizeof(float) * MASK_SIZE);

	float *M;
	cudaMallocManaged(&M, sizeof(float) * MASK_SIZE);
	for (int i = 0; i < MASK_SIZE; i++)
	{
		M[i] = 1.0;
	}
	dim3 blockSize(BLOCK_SIZE);
	dim3 gridSize((blockSize.x + N_Size - 1) / blockSize.x);
	double t1 = getCurrentTime();
	convolution_1D<<<gridSize, blockSize>>>(N, M, P1, N_Size, MASK_SIZE);
	cudaDeviceSynchronize();
	double t2 = getCurrentTime();
	convolution_1D_with_constant_memory<<<gridSize, blockSize>>>(N, P2, N_Size, MASK_SIZE);
	cudaDeviceSynchronize();
	double t3 = getCurrentTime();
	convolution_1D_with_shared_memory<<<gridSize, blockSize>>>(N, P3, N_Size, MASK_SIZE);
	cudaDeviceSynchronize();
	double t4 = getCurrentTime();
	cout << "global memory processing time: " << t2 - t1 << "ms" << endl;
	cout << "constant memory processing time: " << t3 - t2 << "ms" << endl;
	cout << "shared memory processing time: " << t4 - t3 << "ms" << endl;
	for (int i = 0; i < N_Size; i++)
	{
		assert(P1[i] == P2[i]);
		assert(P2[i] == P3[i]);
	}
}