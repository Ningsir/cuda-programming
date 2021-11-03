#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <list>
#include <set>
#include <fstream>
#include <cassert>

#include "common/edge.h"

std::string GetFileExtension(std::string fileName)
{
	if (fileName.find_last_of(".") != std::string::npos)
		return fileName.substr(fileName.find_last_of(".") + 1);
	return "";
}

std::string GetFileNameNoExtension(std::string fileName)
{
	if (fileName.find_last_of(".") != std::string::npos)
		return fileName.substr(0, fileName.find_last_of("."));
	return "";
}

/**
 * @brief 对图进行重新编号，从0开始
 * 
 */
void Renumber(std::string filename)
{
	std::ifstream f(filename, std::ios::in);
	std::string line;
	std::stringstream ss;
	std::map<unsigned, std::list<Edge>> adj_list;
	// {original id, new id}
	std::unordered_map<unsigned, unsigned> node_new_id;
	size_t edge_num = 0;
	unsigned node_count = 0;
	while (std::getline(f, line))
	{
		if (line.substr(0, 1) != "#")
		{
			unsigned source, target;
			ss.str("");
			ss.clear();
			ss << line;
			ss >> source >> target;
			adj_list[source].push_back({target});
			if (node_new_id.count(source) == 0)
			{
				node_new_id[source] = node_count++;
			}
			if (node_new_id.count(target) == 0)
			{
				node_new_id[target] = node_count++;
			}
			edge_num++;
		}
	}
	f.close();
	std::cout << "Vertex Num: " << node_new_id.size() << std::endl;
	std::cout << "Edge Num: " << edge_num << std::endl;
	std::string name = GetFileNameNoExtension(filename);
	std::string extensionName = GetFileExtension(filename);
	std::ofstream outfile(name + "-renumber." + extensionName,
						  std::ofstream::out);
	outfile << "# source\ttarget" << std::endl;
	for (auto &pair : adj_list)
	{
		for (auto &edge : pair.second)
		{
			outfile << node_new_id[pair.first];
			outfile << "\t";
			outfile << node_new_id[edge.target] << std::endl;
		}
	}
	outfile.close();
}

/**
 * @brief 将图转化成二进制格式，节点ID从0开始的图
 * 
 */
void ConvertGraphIdFrom0(std::string filename)
{
	std::ifstream f(filename, std::ios::in);
	std::string line;
	std::stringstream ss;
	std::map<unsigned, std::list<Edge>> adj_list;
	// 用于统计顶点的数量
	std::set<unsigned> set;
	size_t edge_num = 0;
	while (std::getline(f, line))
	{
		if (line.substr(0, 1) != "#")
		{
			unsigned source, target;
			ss.str("");
			ss.clear();
			ss << line;
			ss >> source >> target;
			adj_list[source].push_back({target});
			set.emplace(source);
			set.emplace(target);
			edge_num++;
		}
	}
	f.close();
	std::cout << "Vertex Num: " << set.size() << std::endl;
	std::cout << "Edge Num: " << edge_num << std::endl;
	unsigned *row_ptr_ = new unsigned[set.size() + 1];
	Edge *edge_list_ = new Edge[edge_num];
	size_t edge_count = 0;
	row_ptr_[0] = 0;
	for (unsigned i = 0; i < set.size(); i++)
	{
		if (adj_list.find(i) != adj_list.end())
		{
			auto edge_list = adj_list[i];
			row_ptr_[i + 1] = row_ptr_[i] + edge_list.size();
			for (auto e : edge_list)
			{
				unsigned target = e.target;
				edge_list_[edge_count].target = target;
				edge_count++;
			}
		}
		// 顶点 i 没有出边，顶点i也需要保存在图中
		else
		{
			row_ptr_[i + 1] = row_ptr_[i];
		}
	}
	assert(edge_num == row_ptr_[set.size()]);
	assert(edge_count == edge_num);
	std::string name = GetFileNameNoExtension(filename);
	std::ofstream outfile(name + ".bcsr",
						  std::ofstream::binary);
	unsigned node_num = set.size();
	outfile.write((char *)&node_num, sizeof(unsigned));
	outfile.write((char *)&edge_num, sizeof(size_t));
	outfile.write((char *)row_ptr_, sizeof(unsigned) * (node_num + 1));
	outfile.write((char *)edge_list_, sizeof(Edge) * (edge_count));
	outfile.close();
	delete[] row_ptr_;
	delete[] edge_list_;
}
int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cout << "usage:" << std::endl;
		std::cout << "./convert [renumber | convert] filename" << std::endl;
		exit(0);
	}
	std::string filename = argv[2];
	std::string app = argv[1];
	if (app == "renumber")
	{
		Renumber(filename);
	}
	else if (app == "convert")
	{
		ConvertGraphIdFrom0(filename);
	}
	else
	{
		std::cout << "Error: just using [renumber] or [convert] application." << std::endl;
	}
	return 0;
}