#include <iostream>
#include <assert.h>

#include "utils.h"

using namespace std;

/**
 * @brief matrix multiply, C = A * B.
 * 
 * @param A matrix A,size of M x N
 * @param B matrix B, size of N x K
 * @param[out] C matrix C, size of M x K
 */
__global__ void matrixMulKernel(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int column = blockDim.y * blockIdx.y + threadIdx.y;
	if (row < M && column < K)
	{
		float sum = 0.0;
		for (int i = 0; i < N; i++)
		{
			sum += A[row * N + i] * B[i * K + column];
		}
		C[row * K + column] = sum;
	}
}

#define TILE_WIDTH 8
__global__ void matrixMulKernelWithSharedMem(float *M, float *N, float *P,
											 int Width)
{

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the M and N tiles required to compute P element
	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph)
	{
		// Collaborative loading of M and N tiles into shared memory
		Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
		Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	P[Row * Width + Col] = Pvalue;
}
int main()
{
	int M = 256, N = 256, K = 256;

	float *A, *B, *C, *P;
	cudaMallocManaged(&A, sizeof(float) * M * N);
	cudaMallocManaged(&B, sizeof(float) * N * K);
	cudaMallocManaged(&C, sizeof(float) * M * K);
	cudaMallocManaged(&P, sizeof(float) * M * K);
	for (int i = 0; i < M * N; i++)
	{
		A[i] = rand() * 0.1;
	}
	for (int i = 0; i < N * K; i++)
	{
		B[i] = rand() * 0.1;
	}
	dim3 blockSize(8, 8);
	dim3 gridSize((blockSize.x + M - 1) / blockSize.x, (blockSize.y + K - 1) / blockSize.y);

	double t1 = getCurrentTime();
	matrixMulKernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();
	double t2 = getCurrentTime();
	matrixMulKernelWithSharedMem<<<gridSize, blockSize>>>(A, B, P, M);
	cudaDeviceSynchronize();
	double t3 = getCurrentTime();
	cout << "全局内存处理时间：" << t2 - t1 << " ms" << endl;
	cout << "共享内存处理时间：" << t3 - t2 << " ms" << endl;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < K; j++)
		{
			// cout << C[i * M + j] << ", ";
			assert(C[i * M + j] == P[i * M + j]);
		}
		// cout << endl;
	}
}