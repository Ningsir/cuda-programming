#include <iostream>

struct Matrix
{
    int width;
    int height;
    float *elements;
};

__device__ float getElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}

__device__ void setElement(Matrix *A, int row, int col, float value)
{
    A->elements[row * A->width + col] = value;
}

__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
    float Cvalue = 0.0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < A->width; ++i)
    {
        Cvalue += getElement(A, row, i) * getElement(B, i, col);
    }
    setElement(C, row, col, Cvalue);
}

int main()
{
    int width = 32;
    int height = 32;
    Matrix *A, *B, *C;

    cudaMallocManaged((void **)&A, sizeof(Matrix));
    cudaMallocManaged((void **)&B, sizeof(Matrix));
    cudaMallocManaged((void **)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void **)&A->elements, nBytes);
    cudaMallocManaged((void **)&B->elements, nBytes);
    cudaMallocManaged((void **)&C->elements, nBytes);

    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    matMulKernel<<<gridSize, blockSize>>>(A, B, C);

    cudaDeviceSynchronize();
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            std::cout << C->elements[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "maxError: " << maxError << std::endl;

    return 0;
}
