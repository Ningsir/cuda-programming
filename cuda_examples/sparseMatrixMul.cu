#include <iostream>
#include <vector>
#include <time.h>

#include "common/matrix.h"
#include "common/utils.h"

using namespace std;

template <typename T>
__global__ void sparseMatrixMul(unsigned int *row_ptr, unsigned int *col_index,
								T *w, T *a, T *o, unsigned int m, unsigned int n, unsigned int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	int start = row_ptr[row];
	int end = row_ptr[row + 1];

	T sum = 0;
	for (int i = 0; i < end - start; i++)
	{
		sum += w[start + i] * a[col_index[start + i] * k + col];
	}
	o[row * k + col] = sum;
}

int main()
{
	srand((unsigned)(time(NULL)));
	int m = 80, n = 80, k = 80, max = 100;
	CSR<float> csr;
	GenerateSparseMatrix(csr, m, n, 0.9, max);
	Matrix<float> matrix;
	GenerateMatrix(matrix, n, k, max);

	unsigned *d_row_ptr, *d_column_index;
	float *d_w, *d_a, *d_o;
	cudaMalloc(&d_row_ptr, sizeof(unsigned) * csr.row_num_);
	cudaMalloc(&d_column_index, sizeof(unsigned) * csr.data_num_);
	cudaMalloc(&d_w, sizeof(float) * csr.data_num_);
	cudaMalloc(&d_a, sizeof(float) * matrix.row_ * matrix.column_);
	cudaMalloc(&d_o, sizeof(float) * m * k);

	cudaMemcpy(d_row_ptr, csr.row_ptr, sizeof(unsigned) * csr.row_num_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_column_index, csr.col, sizeof(unsigned) * csr.data_num_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, csr.data, sizeof(unsigned) * csr.data_num_, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, matrix.data_, sizeof(unsigned) * matrix.row_ * matrix.column_, cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);
	dim3 gridSize((csr.row_num_ - 1 + blockSize.x - 1) / blockSize.x,
				  (matrix.column_ + blockSize.y - 1) / blockSize.y);
	sparseMatrixMul<<<gridSize, blockSize>>>(d_row_ptr, d_column_index, d_w, d_a, d_o, m, n, k);
	cudaDeviceSynchronize();

	float *output = (float *)(malloc(sizeof(float) * m * k));
	cudaMemcpy(output, d_o, sizeof(unsigned) * m * k, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m * k; i++)
	{
		cout << output[i] << ", ";
	}
	free(output);
	cudaFree(d_row_ptr);
	cudaFree(d_column_index);
	cudaFree(d_w);
	cudaFree(d_a);
	cudaFree(d_o);
	return 0;
}