#include <iostream>
#include <string>

#include "src/kernel.cuh"
#include "common/graph.h"
#include "common/utils.h"

using namespace std;
int main()
{
	string filename = "/home/xinger/cpp/cuda/cuda-programming/cuda_examples/common/bin/test.txt";
	CSRGraph<float> graph(filename);

	unsigned edge_num = graph.GetEdgeNum();

	// copy data to gpu
	graph.ToDevice();
	// graph.OutputGraph();
	unsigned *d_result;
	unsigned result = 0;
	cudaMalloc(&d_result, sizeof(unsigned));
	cudaMemcpy(d_result, &result, sizeof(unsigned), cudaMemcpyHostToDevice);
	dim3 blockSize(256);
	dim3 gridSize((edge_num + blockSize.x - 1) / blockSize.x);
	double t1 = getCurrentTime();
	// launch kernel
	triangleCountLaunch(graph, d_result, gridSize, blockSize);
	cudaDeviceSynchronize();
	double t2 = getCurrentTime();
	cout << "time: " << t2 - t1 << " ms" << endl;
	// unsigned result;
	cudaMemcpy(&result, d_result, sizeof(unsigned), cudaMemcpyDeviceToHost);

	cout << "triangle count: " << result / 6 << endl;
	cudaFree(d_result);
}