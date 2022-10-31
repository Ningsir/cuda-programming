#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <set>
#include <cassert>
#include <cuda.h>

#include "edge.h"

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
                std::cout << i << "-->" << edge_list_[j].end << std::endl;
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
    inline unsigned GetEdgeNum() const
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
    void init(unsigned node_num, unsigned edge_num)
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
    unsigned edge_num_;
};

/**
 * @brief 对无权重图进行初始化，顶点编号0~n
 */
template <>
void CSRGraph<OutEdge>::InitGraph()
{
    std::ifstream f(filename_, std::ifstream::binary);
    f.read((char *)&node_num_, sizeof(unsigned));
    f.read((char *)&edge_num_, sizeof(unsigned));
    init(node_num_, edge_num_);
    f.read((char *)row_ptr_, sizeof(unsigned) * (node_num_ + 1));
    f.read((char *)edge_list_, sizeof(OutEdge) * (edge_num_));
    std::cout << "Vertex Num: " << node_num_ << std::endl;
    std::cout << "Edge Num: " << edge_num_ << std::endl;
    f.close();
}

/**
 * @brief 对权重图进行初始化
 */
template <>
void CSRGraph<OutEdgeWeighted>::InitGraph()
{
    std::ifstream f(filename_, std::ifstream::binary);
    f.read((char *)&node_num_, sizeof(unsigned));
    f.read((char *)&edge_num_, sizeof(unsigned));
    init(node_num_, edge_num_);
    f.read((char *)row_ptr_, sizeof(unsigned) * (node_num_ + 1));
    f.read((char *)edge_list_, sizeof(OutEdgeWeighted) * (edge_num_));
    std::cout << "Vertex Num: " << node_num_ << std::endl;
    std::cout << "Edge Num: " << edge_num_ << std::endl;
    f.close();
}
