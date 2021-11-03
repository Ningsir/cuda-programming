#include <string>

#include "common/cuda_helper.cuh"
#include "common/graph.h"
#include "src/kernel.cuh"

using namespace std;

int main()
{
	unsigned source = 0;
	std::string output = "./output.txt";
	string filename = "./data/test-renumber.bcsr";
	CSRGraph<Edge> graph(filename);
	graph.OutputGraph();
	graph.ToDevice();
	// 活跃顶点
	unsigned *active_node = (unsigned *)(malloc(sizeof(unsigned) * graph.GetNodeNum()));
	bool *visited = (bool *)(malloc(sizeof(bool) * graph.GetNodeNum()));
	// 用于判断顶点在下一次迭代中是否为活跃的，从而得到活跃顶点
	bool *next_is_active = (bool *)(malloc(sizeof(bool) * graph.GetNodeNum()));
	// bool *finished = (bool *)(malloc(sizeof(bool)));
	// 初始迭代只有一个活跃顶点
	active_node[0] = source;
	memset(visited, 0, sizeof(bool) * graph.GetNodeNum());
	memset(next_is_active, 0, sizeof(bool) * graph.GetNodeNum());
	// *finished = false;

	unsigned *d_active_node;
	bool *d_visited;
	bool *d_next_is_active;
	// bool *d_finished;

	cudaMalloc(&d_active_node, sizeof(unsigned) * graph.GetNodeNum());
	cudaMalloc(&d_visited, sizeof(bool) * graph.GetNodeNum());
	cudaMalloc(&d_next_is_active, sizeof(bool) * graph.GetNodeNum());
	// cudaMalloc(&d_finished, sizeof(bool));

	//copy
	cudaMemcpy(d_active_node, active_node, sizeof(unsigned) * graph.GetNodeNum(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_visited, visited, sizeof(bool) * graph.GetNodeNum(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_next_is_active, next_is_active, sizeof(bool) * graph.GetNodeNum(), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_finished, finished, sizeof(bool), cudaMemcpyHostToDevice);

	unsigned *res;
	cudaMallocManaged(&res, sizeof(unsigned) * graph.GetNodeNum());
	// 结果初始化
	for (int i = 0; i < graph.GetNodeNum(); i++)
	{
		res[i] = std::numeric_limits<unsigned int>::max() - 1;
	}
	res[source] = 0;

	unsigned active_node_num = 1;
	unsigned iter = 0;
	double active_time = 0.0;
	double kernel_time = 0.0;
	double t1 = getCurrentTime();
	while (active_node_num > 0)
	{
		// *finished = true;
		// cudaMemcpy(d_finished, finished, sizeof(bool), cudaMemcpyHostToDevice);
		double k_time = getCurrentTime();
		bfsSyncKernel<<<(active_node_num + 255) / 256, 256>>>(d_active_node,
															  graph.GetDeviceRowPtr(),
															  graph.GetDeviceEdgeList(),
															  d_visited,
															  d_next_is_active,
															  res,
															  active_node_num);
		cudaDeviceSynchronize();
		kernel_time += getCurrentTime() - k_time;
		checkCudaErrors(cudaPeekAtLastError());

		cudaMemcpy(active_node, d_active_node, sizeof(unsigned) * graph.GetNodeNum(), cudaMemcpyDeviceToHost);
		cudaMemcpy(next_is_active, d_next_is_active, sizeof(bool) * graph.GetNodeNum(), cudaMemcpyDeviceToHost);
		// cudaMemcpy(finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		double t3 = getCurrentTime();
		// 更新活跃顶点
		unsigned count = 0;
		for (int i = 0; i < graph.GetNodeNum(); i++)
		{
			if (next_is_active[i])
			{
				active_node[count] = i;
				count++;
			}
		}
		memset(next_is_active, 0, sizeof(bool) * graph.GetNodeNum());
		cudaMemcpy(d_active_node, active_node, sizeof(unsigned) * graph.GetNodeNum(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_next_is_active, next_is_active, sizeof(bool) * graph.GetNodeNum(), cudaMemcpyHostToDevice);

		active_time += getCurrentTime() - t3;
		std::cout << "iter " << iter << ", active node: " << count << std::endl;
		active_node_num = count;

		iter++;
	}
	double t2 = getCurrentTime();
	std::cout << "compute time: " << t2 - t1 << std::endl;
	std::cout << "active time: " << active_time << std::endl;
	std::cout << "kernel time: " << kernel_time << std::endl;
	std::cout << "iter num: " << iter << std::endl;
	SaveResults(output, res, graph.GetNodeNum());

	free(active_node);
	free(next_is_active);
	free(visited);
	// free(finished);
	cudaFree(d_active_node);
	cudaFree(d_next_is_active);
	cudaFree(d_visited);
	// cudaFree(d_finished);
	return 0;
}