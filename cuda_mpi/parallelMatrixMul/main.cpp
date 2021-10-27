#include <iostream>
#include <cuda.h>
#include <mpi.h>

#include "matrix.cuh"

using namespace std;
int main()
{
	MPI_Init(NULL, NULL);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int M = 256, N = 256, K = 256;
	float *A, *B, *C;
	A = (float *)malloc(sizeof(float) * M * N);
	B = (float *)malloc(sizeof(float) * N * K);
	C = (float *)malloc(sizeof(float) * M * K);

	// communication
	if (rank == 0)
	{
		for (int i = 0; i < M * N; i++)
		{
			A[i] = rand() * 0.1;
		}
		for (int i = 0; i < N * K; i++)
		{
			B[i] = rand() * 0.1;
		}
		cout << "rank" << rank << " send message" << endl;
		MPI_Send(A, M * N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(B, N * K, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
		cout << "rank" << rank << " send finished" << endl;
		// launch(A, B, C, M, N, K);
	}
	// computing
	else
	{
		cout << "rank" << rank << " recieve message" << endl;
		MPI_Recv(A, M * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(B, N * K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cout << "rank" << rank << " receive finished" << endl;
		launch(A, B, C, M, N, K);
		cout << "rank" << rank << " send result" << endl;
		MPI_Send(C, M * K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		cout << "rank" << rank << " send result finished" << endl;
	}

	if (rank == 0)
	{
		cout << "rank" << rank << " receive result" << endl;
		MPI_Recv(C, M * K, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cout << "rank" << rank << " receive result fnished" << endl;
	}
	free(A);
	free(B);
	free(C);
	MPI_Finalize();
	return 0;
}
