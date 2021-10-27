#ifndef CUDA_MPI_MATRIX_H_
#define CUDA_MPI_MATRIX_H_

#include <cuda.h>

// __global__ void matrixMulKernel(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K);

void launch(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K);

#endif // CUDA_MPI_MATRIX_H_
