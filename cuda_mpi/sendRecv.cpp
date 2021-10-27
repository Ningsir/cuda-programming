#include <iostream>

#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
	// 初始化 MPI 环境
	MPI_Init(NULL, NULL);

	// 通过调用以下方法来得到所有可以工作的进程数量
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// 得到当前进程的秩，也就是进程号
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// 点对点通信
	// send message
	if (world_rank == 0)
	{
		// int msg = 100;
		// MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		cout << "no messages" << endl;
	}
	// receive message
	else
	{
		int msg;
		MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cout << "message: " << msg << endl;
	}

	// 释放 MPI 的一些资源
	MPI_Finalize();
}