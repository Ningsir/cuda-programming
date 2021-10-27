#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <cassert>

#include "common/utils.h"

#define ARRAY_SIZE 1000000
#define TOTAL_NUMBER 64

using namespace std;

__global__ void histogramKernel_0(unsigned char *d_hist_data, unsigned int *d_bin_data)
{
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_idx < ARRAY_SIZE)
	{
		unsigned char value = d_hist_data[thread_idx];
		atomicAdd(&(d_bin_data[value]), 1);
	}
}

__shared__ unsigned int d_bin_data_shared[TOTAL_NUMBER];
/**
 * @brief histogram computing with shared memory
 */
__global__ void histogramKernel_1(unsigned char *d_hist_data, unsigned int *d_bin_data)
{
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_bin_data_shared[threadIdx.x] = 0;
	__syncthreads();
	if (thread_idx < ARRAY_SIZE)
	{
		unsigned char value = d_hist_data[thread_idx];
		atomicAdd(&(d_bin_data_shared[value]), 1);
		__syncthreads();
		atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
	}
}

int main()
{
	unsigned char h_data[ARRAY_SIZE];
	// array init
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		h_data[i] = rand() % TOTAL_NUMBER;
	}
	unsigned char *d_data;
	cudaMalloc(&d_data, sizeof(unsigned char) * ARRAY_SIZE);
	cudaMemcpy(d_data, h_data, sizeof(unsigned char) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	unsigned int *result1;
	cudaMallocManaged(&result1, sizeof(unsigned int) * TOTAL_NUMBER);
	memset(result1, 0, sizeof(unsigned int) * TOTAL_NUMBER);
	dim3 blockSize(TOTAL_NUMBER);
	dim3 gridSize((blockSize.x + ARRAY_SIZE - 1) / blockSize.x);

	double t1 = getCurrentTime();
	histogramKernel_0<<<gridSize, blockSize>>>(d_data, result1);
	cudaDeviceSynchronize();
	double t2 = getCurrentTime();
	cout << "全局内存处理时间：" << t2 - t1 << " ms" << endl;

	unsigned int *result2;
	cudaMallocManaged(&result2, sizeof(unsigned int) * TOTAL_NUMBER);
	double t3 = getCurrentTime();
	histogramKernel_1<<<gridSize, blockSize>>>(d_data, result2);
	cudaDeviceSynchronize();
	double t4 = getCurrentTime();

	cout << "共享内存处理时间：" << t4 - t3 << " ms" << endl;
	for (int i = 0; i < TOTAL_NUMBER; i++)
	{
		// cout << (result1[i] == result2[i] ? 1 : i) << endl;
		// sum += result2[i];
		assert(result1[i] == result2[i]);
	}
	cudaFree(d_data);
	return 0;
}