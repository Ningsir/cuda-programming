#include <iostream>
#include <vector>
using namespace std;

template <typename T>
struct CSR
{
	// row_ptr[i + 1] - row_ptr[i]表示第i行非0元素的个数
	std::unique_ptr<unsigned int> row_ptr;
	// unsigned int *row_ptr;
	std::unique_ptr<unsigned int> col;
	unsigned int row_num;
	unsigned int *col;
	T *data;
	unsigned int data_num;
	~CSR()
	{
	}
};

/**
 * @brief convert matrix to CSR format.
 * 
 * @param matrix a matrix of size m x n.
 */
template <typename T>
void Matrix2CSR(const T *matrix, CSR<T> &csr, unsigned int m, unsigned int n)
{
}
__global__ void sparseMatrixMul()
{
}