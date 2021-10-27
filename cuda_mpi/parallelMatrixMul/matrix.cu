#include "matrix.cuh"

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

void launch(float *A, float *B, float *Output, unsigned int M, unsigned int N, unsigned int K)
{
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, sizeof(float) * M * N);
	cudaMalloc(&d_B, sizeof(float) * N * K);
	cudaMalloc(&d_C, sizeof(float) * M * K);
	cudaMemcpy(d_A, A, M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N * K, cudaMemcpyHostToDevice);

	dim3 blockSize(8, 8);
	dim3 gridSize((blockSize.x + M - 1) / blockSize.x, (blockSize.y + K - 1) / blockSize.y);

	matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
	cudaDeviceSynchronize();
	cudaMemcpy(Output, d_C, M * K, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}