#ifndef CUDA_EXAMPLES_COMMON_EDGE_H_
#define CUDA_EXAMPLES_COMMON_EDGE_H_

struct Edge
{
	unsigned target;
};

struct WeightEdge
{
	unsigned target;
	float data;
};

#endif // CUDA_EXAMPLES_COMMON_EDGE_H_
