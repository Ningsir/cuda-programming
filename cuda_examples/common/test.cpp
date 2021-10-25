#include <iostream>
#include <vector>
#include <cassert>
#include <time.h>

#include "matrix.h"
#include "utils.h"

using namespace std;
int main()
{
	srand((unsigned)(time(NULL)));
	int row = 10000, column = 100;
	int max = 1000;
	CSR<float> csr;
	GenerateSparseMatrix(csr, row, column, 0.9, max);
	for (int i = 0; i < csr.data_num_; i++)
	{
		assert(csr.data[i] < max);
	}
	Matrix<float> matrix;
	GenerateMatrix<float>(matrix, row, column);
	for (int i = 0; i < matrix.row_; i++)
	{
		for (int j = 0; j < matrix.column_; j++)
		{
			assert(matrix.data_[i * column + j] < max);
		}
	}
	return 0;
}