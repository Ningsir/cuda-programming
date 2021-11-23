#include <iostream>
#include <vector>
#include <time.h>

#include "common/matrix.h"
#include "common/utils.h"
#include "common/cuda_helper.cuh"
#include "src/kernel.cuh"

using namespace std;

int main()
{
	srand((unsigned)(time(NULL)));
	unsigned int m = 8000, n = 80, k = 1000, max = 100;
	// matrix mul: w * matrix_a
	// matrix w
	Matrix<float> w(m, n);
	w.InitWithMostZero(0.9, max);
	CSR<float> csr;
	Matrix2CSR(csr, w);
	// matrix a
	Matrix<float> matrix_a(n, k);
	matrix_a.InitWithRandom(max);

	unsigned *d_row_ptr, *d_column_index;
	float *d_w, *d_a, *d_o;
	checkCudaErrors(cudaMalloc(&d_row_ptr, sizeof(unsigned) * csr.row_num_));
	cudaMalloc(&d_column_index, sizeof(unsigned) * csr.data_num_);
	cudaMalloc(&d_w, sizeof(float) * csr.data_num_);
	cudaMalloc(&d_a, sizeof(float) * matrix_a.row_ * matrix_a.column_);
	cudaMalloc(&d_o, sizeof(float) * m * k);

	cudaMemcpy(d_row_ptr, csr.row_ptr, sizeof(unsigned) * csr.row_num_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_column_index, csr.col, sizeof(unsigned) * csr.data_num_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, csr.data, sizeof(unsigned) * csr.data_num_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, matrix_a.data_, sizeof(unsigned) * matrix_a.row_ * matrix_a.column_, cudaMemcpyHostToDevice);

	dim3 blockSize(8, 8);
	dim3 gridSize((csr.row_num_ - 1 + blockSize.x - 1) / blockSize.x,
				  (matrix_a.column_ + blockSize.y - 1) / blockSize.y);
	sparseMatrixMulLaunch<float>(gridSize, blockSize, d_row_ptr, d_column_index, d_w, d_a, d_o, m, n, k);
	// 检查内核参数错误
	checkCudaErrors(cudaPeekAtLastError());
	cudaDeviceSynchronize();

	float *output = (float *)(malloc(sizeof(float) * m * k));
	cudaMemcpy(output, d_o, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

	// dense matrix mul
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, sizeof(float) * m * n);
	cudaMalloc(&d_B, sizeof(float) * n * k);
	cudaMalloc(&d_C, sizeof(float) * m * k);

	cudaMemcpy(d_A, w.data_, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, matrix_a.data_, sizeof(float) * matrix_a.row_ * matrix_a.column_, cudaMemcpyHostToDevice);
	matrixMul(gridSize, blockSize, d_A, d_B, d_C, m, n, k);
	cudaDeviceSynchronize();
	float *output1 = (float *)(malloc(sizeof(float) * m * k));
	cudaMemcpy(output1, d_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

	for (int i = 0; i < m * k; i++)
	{
		assert(output[i] == output1[i]);
	}
	free(output);
	free(output1);
	cudaFree(d_row_ptr);
	cudaFree(d_column_index);
	cudaFree(d_w);
	cudaFree(d_a);
	cudaFree(d_o);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}