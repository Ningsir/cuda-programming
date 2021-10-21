# Introduction

# 卷积

## 全局内存实现

一维卷积：输入数组是一维的，卷积核是一维的，为了方便计算，卷积核的长度为奇数。每个cuda线程负责一个输出元素的计算。

```cpp
__global__ void convolution_1D(float *N, float *M, float *P, unsigned int N_Size, unsigned int Mask_Size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;
	int left = id - Mask_Size / 2;
	for (int i = 0; i < Mask_Size; i++)
	{
		if (left + i < N_Size && left + i >= 0)
			sum += M[i] * N[left + i];
	}
	P[id] = sum;
}

// dim3 blockSize(256);
// dim3 gridSize((blockSize.x + N_Size - 1) / blockSize.x);
// convolution_1D<<<gridSize, blockSize>>>(N, M, P1, N_Size, MASK_SIZE);
```

问题：

1. 控制流多样性：卷积操作在计算边界时，会有不同的处理方式，会导致if有不同的处理方式，也就导致不同线程的开销不一样。对于大型数组和小型卷积核，控制流多样性带来的影响很小。
2. 存储器带宽：浮点运算和全局内存访问的比率为`1:1`，严重影响性能，需要减少全局内存的访问。

## 常量内存

# Sparse Matrix Multiply

# 参考
1. [矩阵乘法优化](https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html)