#ifndef CUDA_EXAMPLES_COMMON_GRAPH_H_
#define CUDA_EXAMPLES_COMMON_GRAPH_H_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <set>
#include <cassert>
#include <cuda.h>

#include "matrix.h"
#include "common/edge.h"

/**
 * The graph is represtented in CSR format and COO format.
 * The template param T represents the type of edge data.
 */
template <typename T>
class CSRCOOGraph
{

public:
	CSRCOOGraph(std::string filename, bool directed = true) : filename_(std::move(filename)),
															  directed_(directed)
	{
		initGraph();
	}
	~CSRCOOGraph()
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
 	* @brief 文件中的顶点id从 1 ~ n，我们需要将其转换成 0 ~ n-1
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
					// 因为文件中的顶点编号从1~n，将其转换成0~n-1
					csr.col[edge_count] = target - 1;
					csr.data[edge_count] = data;
					edge_count++;
				}
			}
			// 顶点 i 没有出边，顶点i也需要保存在图中
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
	unsigned *row_id;

	unsigned *d_row_ptr;
	// (row_id, col_id) represents COO matrix.
	unsigned *d_row_id;
	unsigned *d_col_id;
};

// The graph is represtented in CSR format.
template <typename E>
class CSRGraph
{
public:
	CSRGraph(std::string filename) : filename_(std::move(filename))
	{
		InitGraph();
	}

	void InitGraph();

	void ToDevice()
	{
		has_to_device_ = true;
		cudaMalloc(&d_row_ptr_, sizeof(unsigned) * (node_num_ + 1));
		cudaMalloc(&d_edge_list_, sizeof(E) * edge_num_);

		cudaMemcpy(d_row_ptr_, row_ptr_, sizeof(unsigned) * (node_num_ + 1), cudaMemcpyHostToDevice);
		cudaMemcpy(d_edge_list_, edge_list_, sizeof(E) * edge_num_, cudaMemcpyHostToDevice);
	}
	void OutputGraph()
	{
		for (unsigned i = 0; i < node_num_; i++)
		{
			unsigned start = row_ptr_[i];
			unsigned end = row_ptr_[i + 1];
			for (int j = start; j < end; j++)
			{
				std::cout << i << "-->" << edge_list_[j].target << std::endl;
			}
		}
	}
	inline unsigned *GetDeviceRowPtr() const
	{
		return d_row_ptr_;
	}

	inline E *GetDeviceEdgeList() const
	{
		return d_edge_list_;
	}
	inline unsigned *GetRowPtr() const
	{
		return row_ptr_;
	}

	inline E *GetEdgeList() const
	{
		return edge_list_;
	}
	inline size_t GetEdgeNum() const
	{
		return edge_num_;
	}
	inline unsigned GetNodeNum() const
	{
		return node_num_;
	}
	~CSRGraph()
	{
		delete[] row_ptr_;
		delete[] edge_list_;
		if (has_to_device_)
		{
			cudaFree(d_row_ptr_);
			cudaFree(d_edge_list_);
		}
	}

private:
	void init(unsigned node_num, size_t edge_num)
	{
		node_num_ = node_num;
		edge_num_ = edge_num;
		row_ptr_ = new unsigned[node_num_ + 1];
		edge_list_ = new E[edge_num_];
	}
	std::string filename_;
	// 长度为node_num_ + 1
	unsigned *row_ptr_;
	// 长度为edge_num_
	E *edge_list_;

	// 是否已经将数据copy到gpu内存中
	bool has_to_device_ = false;
	unsigned *d_row_ptr_;
	E *d_edge_list_;
	unsigned node_num_;
	size_t edge_num_;
};

/**
 * @brief 对无权重图进行初始化，顶点编号0~n
 */
template <>
void CSRGraph<Edge>::InitGraph()
{
	std::ifstream f(filename_, std::ifstream::binary);
	f.read((char *)&node_num_, sizeof(unsigned));
	f.read((char *)&edge_num_, sizeof(size_t));
	init(node_num_, edge_num_);
	f.read((char *)row_ptr_, sizeof(unsigned) * (node_num_ + 1));
	f.read((char *)edge_list_, sizeof(Edge) * (edge_num_));
	std::cout << "Vertex Num: " << node_num_ << std::endl;
	std::cout << "Edge Num: " << edge_num_ << std::endl;
	f.close();
	// std::string line;
	// std::stringstream ss;
	// std::map<unsigned, std::list<Edge>> adj_list;
	// // 用于统计顶点的数量
	// std::set<unsigned> set;
	// size_t edge_num = 0;
	// while (std::getline(f, line))
	// {
	// 	unsigned source, target;
	// 	ss.str("");
	// 	ss.clear();
	// 	ss << line;
	// 	ss >> source >> target;
	// 	adj_list[source].push_back({target});
	// 	set.emplace(source);
	// 	set.emplace(target);
	// 	edge_num++;
	// }
	// f.close();
	// std::cout << "Vertex Num: " << set.size() << std::endl;
	// std::cout << "Edge Num: " << edge_num << std::endl;
	// init(set.size(), edge_num);
	// size_t edge_count = 0;
	// row_ptr_[0] = 0;
	// for (unsigned i = 0; i < set.size(); i++)
	// {
	// 	if (adj_list.find(i) != adj_list.end())
	// 	{
	// 		auto edge_list = adj_list[i];
	// 		row_ptr_[i + 1] = row_ptr_[i] + edge_list.size();
	// 		for (auto e : edge_list)
	// 		{
	// 			unsigned target = e.target;
	// 			// 因为文件中的顶点编号从1~n，将其转换成0~n-1
	// 			edge_list_[edge_count].target = target;
	// 			edge_count++;
	// 		}
	// 	}
	// 	// 顶点 i 没有出边，顶点i也需要保存在图中
	// 	else
	// 	{
	// 		row_ptr_[i + 1] = row_ptr_[i];
	// 	}
	// }
	// assert(edge_num == row_ptr_[node_num_]);
	// assert(edge_count == edge_num);
}

/**
 * @brief 对权重图进行初始化
 */
template <>
void CSRGraph<WeightEdge>::InitGraph()
{
}

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
