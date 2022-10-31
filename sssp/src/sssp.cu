#include <limits>
#include <string>
#include <sys/time.h>
#include <cuda.h>

#include "graph.h"

using namespace std;

#define checkCudaErrors(ans)                  \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

double getCurrentTime()
{
    timeval t;
    gettimeofday(&t, nullptr);
    return static_cast<double>(t.tv_sec) * 1000 +
           static_cast<double>(t.tv_usec) / 1000;
}

void SaveResults(std::string filepath, int *results, uint n)
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

__global__ void sssp_kernel(unsigned *row_ptr, OutEdgeWeighted *edge_list, int *res, bool *finished, int node_nums)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < node_nums)
    {
        unsigned start = row_ptr[tid];
        unsigned end = row_ptr[tid + 1];
        int sourceDist = res[tid];
        unsigned target;
        int targetDist;
        // traverse the neighbours
        for (int i = start; i < end; i++)
        {
            target = edge_list[i].end;
            targetDist = sourceDist + edge_list[i].w8;
            if (targetDist < res[target])
            {
                atomicMin(&res[target], targetDist);
                // printf("source: %d, dis:%d ;target:%d, dis: %d\n", tid, sourceDist, target, targetDist);
                *finished = false;
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "\nThere was an error parsing command line arguments\n";
        exit(0);
    }
    unsigned source = atoi(argv[1]);
    string filename = string(argv[2]);
    std::string output = "./output.txt";
    // string filename = "./data/test.bwcsr";
    CSRGraph<OutEdgeWeighted> graph(filename);
    graph.OutputGraph();
    graph.ToDevice();
    unsigned node_num = graph.GetNodeNum();
    int *res;
    cudaMallocManaged(&res, sizeof(int) * graph.GetNodeNum());
    // 结果初始化
    for (int i = 0; i < graph.GetNodeNum(); i++)
    {
        res[i] = 100000;
    }
    res[source] = 0;
    unsigned iter = 0;
    bool finished = false;
    bool *d_finished;
    cudaMalloc(&d_finished, sizeof(bool));
    cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice);
    double t1 = getCurrentTime();
    while (!finished)
    {
        double k_time = getCurrentTime();
        sssp_kernel<<<(node_num + 255) / 256, 256>>>(
            graph.GetDeviceRowPtr(),
            graph.GetDeviceEdgeList(),
            res,
            d_finished,
            node_num);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaPeekAtLastError());
        cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_finished, true, sizeof(bool));
        iter++;
    }
    double t2 = getCurrentTime();
    std::cout << "compute time: " << t2 - t1 << " ms" << std::endl;
    std::cout << "iter num: " << iter << std::endl;
    SaveResults(output, res, graph.GetNodeNum());
    cudaFree(d_finished);
    return 0;
}
