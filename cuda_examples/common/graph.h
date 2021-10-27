#ifndef CUDA_EXAMPLES_COMMON_GRAPH_H_
#define CUDA_EXAMPLES_COMMON_GRAPH_H_

#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <cassert>
#include <cuda.h>

#include "matrix.h"

/**
 * The graph is represtented in CSR format.
 * The template param T represents the type of edge data.
 */
template <typename T>
class CSRGraph
{

public:
	CSRGraph(std::string filename, bool directed = true) : filename_(std::move(filename)),
														   directed_(directed)
	{
		initGraph();
	}
	~CSRGraph()
	{
		if (d_row_ptr != NULL)
		{
			cudaFree(d_row_ptr);
			cudaFree(d_row_id);
			cudaFree(d_col_id);
		}
	}

	void ToDevice()
	{
		cudaMalloc(&d_row_ptr, sizeof(unsigned) * csr.row_num_);
		cudaMalloc(&d_row_id, sizeof(unsigned) * csr.data_num_);
		cudaMalloc(&d_col_id, sizeof(unsigned) * csr.data_num_);
		cudaMemcpy(d_row_ptr, csr.row_ptr, sizeof(unsigned) * csr.row_num_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_col_id, csr.col, sizeof(unsigned) * csr.data_num_, cudaMemcpyHostToDevice);

		unsigned *row_id = (unsigned *)malloc(sizeof(unsigned) * csr.data_num_);
		for (unsigned i = 0; i < csr.row_num_ - 1; i++)
		{
			unsigned start = csr.row_ptr[i];
			unsigned end = csr.row_ptr[i + 1];
			for (int j = start; j < end; j++)
			{
				row_id[j] = i;
			}
		}
		// for (unsigned i = 0; i < csr.data_num_; i++)
		// {
		// 	std::cout << row_id[i] << ", ";
		// }
		cudaMemcpy(d_row_id, row_id, sizeof(unsigned) * csr.data_num_, cudaMemcpyHostToDevice);
		free(row_id);
	}

	void OutputGraph()
	{
		for (unsigned i = 0; i < csr.row_num_ - 1; i++)
		{
			unsigned start = csr.row_ptr[i];
			unsigned end = csr.row_ptr[i + 1];
			for (int j = start; j < end; j++)
			{
				std::cout << i << "-->" << csr.col[j] << std::endl;
			}
		}
	}

	inline unsigned *GetRowPtr() const
	{
		return d_row_ptr;
	}

	inline unsigned *GetRowId() const
	{
		return d_row_id;
	}

	inline unsigned *GetColId() const
	{
		return d_col_id;
	}

	inline unsigned GetEdgeNum() const
	{
		return csr.data_num_;
	}

private:
	/**
 	* @brief 顶点序号1-n.
 	*/
	void initGraph()
	{
		std::ifstream f(filename_, std::ios::in);
		std::string line;
		std::stringstream ss;
		std::map<unsigned, std::list<std::pair<unsigned, T>>> adj_list;
		// 用于统计顶点的数量
		std::set<unsigned> set;
		size_t edge_num = 0;
		while (std::getline(f, line))
		{
			unsigned source, target;
			T data;
			ss.str("");
			ss.clear();
			ss << line;
			ss >> source >> target >> data;
			if (directed_)
			{
				adj_list[source].push_back({target, data});
			}
			set.emplace(source);
			set.emplace(target);
			edge_num++;
		}
		f.close();
		std::cout << "Vertex Num: " << set.size() << std::endl;
		std::cout << "Edge Num: " << edge_num << std::endl;
		csr = std::move(CSR<T>(set.size() + 1, edge_num));
		size_t edge_count = 0;
		csr.row_ptr[0] = 0;
		for (unsigned i = 1; i <= set.size(); i++)
		{
			if (adj_list.find(i) != adj_list.end())
			{
				auto edge_list = adj_list[i];
				csr.row_ptr[i] = csr.row_ptr[i - 1] + edge_list.size();
				for (auto pair : edge_list)
				{
					unsigned target = pair.first;
					T data = pair.second;
					csr.col[edge_count] = target - 1;
					csr.data[edge_count] = data;
					edge_count++;
				}
			}
			// 顶点 i 没有出边
			else
			{
				csr.row_ptr[i] = csr.row_ptr[i - 1];
			}
		}
		assert(csr.data_num_ == csr.row_ptr[csr.row_num_ - 1]);
		assert(edge_count == edge_num);
	}
	std::string filename_;
	bool directed_;
	CSR<T> csr;
	unsigned *d_row_ptr;
	unsigned *d_row_id;
	unsigned *d_col_id;
};

// The graph is represtented in CSC format.
template <typename T>
class CSCGraph
{
};
// The graph is represtented in COO format.
template <typename T>
class COOGraph
{
};
#endif // CUDA_EXAMPLES_COMMON_GRAPH_H_
