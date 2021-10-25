#ifndef CUDA_EXAMPLES_COMMON_UTILS_H_
#define CUDA_EXAMPLES_COMMON_UTILS_H_
#include <cassert>

#include "matrix.h"
/**
 * @brief convert matrix to CSR format.
 * 
 * @param matrix a matrix of size m x n.
 */
template <typename T>
void Matrix2CSR(CSR<T> &csr, T *matrix, unsigned int m, unsigned int n)
{
	int total = 0;
	// 统计矩阵中非0元素的个数
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (matrix[i * n + j] != 0)
			{
				total += 1;
			}
		}
	}
	csr = std::move(CSR<T>(m + 1, total));
	total = 0;
	csr.row_ptr[0] = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (matrix[i * n + j] != 0)
			{
				csr.col[total] = j;
				csr.data[total] = matrix[i * n + j];
				total++;
			}
		}
		csr.row_ptr[i + 1] = total;
	}
}
template <typename T>
void GenerateSparseMatrix(CSR<T> &csr, int row, int column, float zero_rate, int max = 100)
{
	T *matrix = (T *)malloc(sizeof(T) * row * column);
	int data = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			data = rand() % max;
			matrix[column * i + j] = data < max * zero_rate ? 0 : static_cast<T>(data);
		}
	}
	Matrix2CSR(csr, matrix, row, column);
	assert(csr.data_num_ == csr.row_ptr[csr.row_num_ - 1]);
	free(matrix);
}

template <typename T>
void GenerateMatrix(Matrix<T> &matrix, int row, int column, int max = 1000)
{
	matrix = std::move(Matrix<T>(row, column));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			T data = static_cast<T>(rand() % max);
			// std::cout << "data : " << data << std::endl;
			matrix.data_[i * column + j] = data;
		}
	}
}

#endif // CUDA_EXAMPLES_COMMON_UTILS_H_
