#include <string>

#include "common/cuda_helper.cuh"
#include "common/graph.h"
#include "src/kernel.cuh"

using namespace std;

int main()
{
	string filename = "/home/xinger/cpp/cuda/cuda-programming/cuda_examples/common/bin/test.txt";
	CSRGraph<Edge> graph(filename);
	// graph.OutputGraph();
	graph.ToDevice();
	bfsSyncLaunch(graph, 0, "./output.txt");
	return 0;
}