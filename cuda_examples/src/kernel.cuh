#ifndef CUDA_EXAMPLES_KERNEL_KERNEL_H_
#define CUDA_EXAMPLES_KERNEL_KERNEL_H_

#include <stdio.h>
#include <string>
#include <iostream>
#include <limits>

#include "common/graph.h"
#include "common/cuda_helper.cuh"
#include "common/utils.h"

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

/**
 * @brief launch matrix multiply kernel.
 */
void matrixMul(dim3 grid_size, dim3 block_size, float *A, float *B, float *C,
			   unsigned int M, unsigned int N, unsigned int K)
{
	matrixMulKernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

template <typename T>
__global__ void sparseMatrixMulKernel(unsigned int *row_ptr, unsigned int *col_index,
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

/**
 * @brief launch sparse matrix multiply kernel; 
 */
template <typename T>
void sparseMatrixMulLaunch(dim3 grid_size, dim3 block_size, unsigned int *row_ptr, unsigned int *col_index,
						   T *w, T *a, T *o, unsigned int m, unsigned int n, unsigned int k)
{
	sparseMatrixMulKernel<T><<<grid_size, block_size>>>(row_ptr, col_index, w, a, o, m, n, k);
}

/**
 * @brief edge-centric triangle counting kernel.
 * The graph with <row_ptr, col_id> is represented in CSR format.
 * The graph with <row_id, col_id> is represented in COO format.
 * 
 * @param row_ptr
 * @param row_id
 * @param col_id
 * @param[out] count result
 */
__global__ void triangleCountKernel(unsigned *row_ptr, unsigned *row_id, unsigned *col_id, unsigned edge_num, unsigned *count)
{
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < edge_num)
	{
		// edge: source -> target
		unsigned source = row_id[id];
		unsigned target = col_id[id];
		// printf("id: %d, %d --> %d\n", id, source, target);
		unsigned source_start = row_ptr[source];
		unsigned source_end = row_ptr[source + 1];

		unsigned target_start = row_ptr[target];
		unsigned target_end = row_ptr[target + 1];

		int i = 0, j = 0;
		// linear search
		while (i < source_end - source_start && j < target_end - target_start)
		{
			if (col_id[i + source_start] == col_id[j + target_start])
			{
				atomicAdd(count, 1);
				// printf("edge: %d -- > %d, common neigh: %d, id: %d \n", source, target, col_id[i + source_start], i + source_start);
				i++;
				j++;
			}
			else if (col_id[i + source_start] > col_id[j + target_start])
			{
				j++;
			}
			else
			{
				i++;
			}
		}
	}
}
/**
 * @brief launch triangle count kernel.
 * 
 */
void triangleCountLaunch(const CSRCOOGraph<float> &graph, unsigned *count, dim3 gridSize, dim3 blockSize)
{
	triangleCountKernel<<<gridSize, blockSize>>>(graph.GetRowPtr(), graph.GetRowId(), graph.GetColId(), graph.GetEdgeNum(), count);
	checkCudaErrors(cudaPeekAtLastError());
}

/**
 * @brief bfs
 */
__global__ void bfsSyncKernel(unsigned *active_nodes, unsigned *row_ptr, Edge *edge_list, bool *visited, unsigned *next_is_active, unsigned *res, unsigned active_node_nums)
{
	unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < active_node_nums)
	{
		unsigned source = active_nodes[tid];
		unsigned start = row_ptr[source];
		unsigned end = row_ptr[source + 1];
		unsigned sourceDist = res[source];
		unsigned target;
		unsigned targetDist;
		// traverse the neighbours
		for (int i = start; i < end; i++)
		{
			target = edge_list[i].target;
			targetDist = sourceDist + 1;
			// printf("source: %d, target node: %d, visited: %d, dist: %d\n", source, target, visited[target], res[target]);
			if (!visited[target] && targetDist < res[target])
			{
				atomicMin(&res[target], targetDist);
				visited[target] = true;
				next_is_active[target] = 1;
				// printf("active node: %d\n", target);
				// *finished = false;
			}
		}
	}
}

void bfsSyncLaunch(const CSRGraph<Edge> &graph, unsigned source)
{
}
#endif // CUDA_EXAMPLES_KERNEL_KERNEL_H_
