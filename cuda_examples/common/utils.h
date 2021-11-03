#ifndef CUDA_EXAMPLES_COMMON_UTILS_H_
#define CUDA_EXAMPLES_COMMON_UTILS_H_
#include <cassert>
#include <sys/time.h>
#include <string>
#include <fstream>

#include "matrix.h"

void SaveResults(std::string filepath, uint *results, uint n)
{
	std::cout << "Saving the results into the following file:\n";
	std::cout << ">> " << filepath << std::endl;
	std::ofstream outfile;
	outfile.open(filepath);
	for (int i = 0; i < n; i++)
		outfile << i << " " << results[i] << std::endl;
	outfile.close();
	std::cout << "Done saving.\n";
}
double getCurrentTime()
{
	timeval t;
	gettimeofday(&t, nullptr);
	return static_cast<double>(t.tv_sec) * 1000 +
		   static_cast<double>(t.tv_usec) / 1000;
}

/**
 * @brief convert csr matrix to dense matrix format.
 * 
 * @param[out] matrix result
 * @param[in] csr input csr matrix
 */
template <typename T>
void CSR2Matrix(Matrix<T> &matrix, CSR<T> &csr)
{
}
/**
 * @brief convert dense matrix to CSR format.
 * 
 * @param matrix a matrix of size m x n.
 */
template <typename T>
void Matrix2CSR(CSR<T> &csr, Matrix<T> &matrix)
{
	int row = matrix.row_;
	int column = matrix.column_;
	int total = 0;
	// 统计矩阵中非0元素的个数
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			if (matrix.data_[i * column + j] != 0)
			{
				total += 1;
			}
		}
	}
	csr = std::move(CSR<T>(row + 1, total));
	total = 0;
	csr.row_ptr[0] = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			if (matrix.data_[i * column + j] != 0)
			{
				csr.col[total] = j;
				csr.data[total] = matrix.data_[i * column + j];
				total++;
			}
		}
		csr.row_ptr[i + 1] = total;
	}
}
template <typename T>
void GenerateSparseMatrix(CSR<T> &csr, int row, int column, float zero_rate, int max = 100)
{
	Matrix<T> matrix(row, column);
	int data = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			data = rand() % max;
			matrix.data_[column * i + j] = data < max * zero_rate ? 0 : static_cast<T>(data);
		}
	}
	Matrix2CSR(csr, matrix);
	assert(csr.data_num_ == csr.row_ptr[csr.row_num_ - 1]);
}

#endif // CUDA_EXAMPLES_COMMON_UTILS_H_
