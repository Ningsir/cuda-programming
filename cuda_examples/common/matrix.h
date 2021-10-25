#ifndef CUDA_EXAMPLES_COMMON_MATRIX_H
#define CUDA_EXAMPLES_COMMON_MATRIX_H

#include <cstring>

template <typename T>
struct Matrix
{
	T *data_;
	unsigned row_;
	unsigned column_;

	Matrix()
	{
		data_ = NULL;
		row_ = 0;
		column_ = 0;
	}
	Matrix(unsigned row, unsigned column)
	{
		row_ = row;
		column_ = column;
		data_ = (T *)malloc(sizeof(T) * row * column);
	}
	Matrix(const Matrix &matrix)
	{
		this->row_ = matrix.row_;
		this->column_ = matrix.column_;
		data_ = (T *)malloc(sizeof(T) * matrix.row_ * matrix.column_);
		// 深拷贝
		memcpy(data_, matrix.data_, sizeof(T) * matrix.row_ * matrix.column_);
	}

	Matrix &operator=(const Matrix &matrix)
	{
		if (this == &matrix)
		{
			return *this;
		}
		if (this->data_ != NULL)
		{
			free(data_);
		}
		data_ = (T *)malloc(sizeof(T) * matrix.row_ * matrix.column_);
		// 深拷贝
		memcpy(data_, matrix.data_, sizeof(T) * matrix.row_ * matrix.column_);
		this->column_ = matrix.column_;
		this->row_ = matrix.row_;
		return *this;
	}

	Matrix &operator=(Matrix &&matrix)
	{
		this->data_ = matrix.data_;
		this->row_ = matrix.row_;
		this->column_ = matrix.column_;
		// 置为NULL，避免被free
		matrix.data_ = NULL;
		return *this;
	}

	~Matrix()
	{
		if (data_ != NULL)
		{
			free(data_);
		}
	}
};
template <typename T>
struct CSR
{
	// row_ptr[i + 1] - row_ptr[i]表示第i行非0元素的个数
	unsigned int *row_ptr;
	// row_ptr数组长度
	unsigned int row_num_;
	unsigned int *col;
	T *data;
	unsigned int data_num_;

	CSR()
	{
	}
	CSR(unsigned int row, unsigned int data_num)
	{
		row_num_ = row;
		data_num_ = data_num;
		row_ptr = (unsigned int *)malloc(sizeof(unsigned int) * (row));
		col = (unsigned int *)malloc(sizeof(unsigned int) * data_num);
		data = (T *)malloc(sizeof(T) * data_num);
	}

	CSR(const CSR &csr)
	{
		row_ptr = (unsigned *)malloc(sizeof(unsigned) * csr.row_num_);
		col = (unsigned *)malloc(sizeof(unsigned) * csr.data_num_);
		data = (T *)malloc(sizeof(T) * csr.data_num_);

		memcpy(row_ptr, csr.row_ptr, sizeof(unsigned) * csr.row_num_);
		memcpy(col, csr.col, sizeof(unsigned) * csr.data_num_);
		memcpy(data, csr.data, sizeof(T) * csr.data_num_);
		row_num_ = csr.row_num_;
		data_num_ = csr.data_num_;
	}

	CSR &operator=(const CSR &csr)
	{
		if (*this == csr)
		{
			return *this;
		}
		// 释放旧空间
		if (row_ptr != NULL)
		{
			free(row_ptr);
			free(col);
			free(data);
		}
		// 分配新空间
		row_ptr = (unsigned *)malloc(sizeof(unsigned) * csr.row_num_);
		col = (unsigned *)malloc(sizeof(unsigned) * csr.data_num_);
		data = (T *)malloc(sizeof(T) * csr.data_num_);
		// 深拷贝
		memcpy(row_ptr, csr.row_ptr, sizeof(unsigned) * csr.row_num_);
		memcpy(col, csr.col, sizeof(unsigned) * csr.data_num_);
		memcpy(data, csr.data, sizeof(T) * csr.data_num_);
		row_num_ = csr.row_num_;
		data_num_ = csr.data_num_;
		return *this;
	}

	CSR &operator=(CSR &&csr)
	{
		row_ptr = csr.row_ptr;
		col = csr.col;
		data = csr.data;
		row_num_ = csr.row_num_;
		data_num_ = csr.data_num_;
		csr.row_ptr = NULL;
		csr.col = NULL;
		csr.data = NULL;
		return *this;
	}
	~CSR()
	{
		if (row_ptr != NULL)
		{
			free(row_ptr);
			free(col);
			free(data);
		}
	}
};

#endif // CUDA_EXAMPLES_COMMON_MATRIX_H
