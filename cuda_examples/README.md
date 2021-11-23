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

# Triangle Count

**分布式实现Triangle Count**的问题：
1. 是每次执行程序的时候都partition还是partition之后存到文件中然后分布式执行呢？？
2. 如果存到本地文件中，还需要确定哪些节点是外部节点？
3. 计算的时候执行通信开销太大，如何避免通信开销呢？
4. 通信是CPU之间通信还是直接GPU之间进行通信呢？

# Reduce

> 访存占主导的一种算子。

[reduction优化](https%3A//developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

reduce是一个多对一的运算，对于一个求和的reduce操作：`sum = lst[0] + lst[1] + ... + lst[n-1]`。
如下图所示，reduce一般分为两个阶段，在第一阶段中，开启m个block计算出m个小份的reduce值。最后，在第二阶段中，使用一个block将m个小份再次进行reduce，得到最终的结果，第二阶段也可以使用原子操作，但是如果数据过大则性能不佳。
![image-20211123095020788](https://gitee.com/huster_ning/image/raw/master//image/image-20211123095020788.png)


## baseline

每个线程块负责一部分数据的reduce操作。
![image-20211123113939028](https://gitee.com/huster_ning/image/raw/master//image/image-20211123113939028.png)
```cpp

#define BLOCK_SIZE 256
__global__ void sum(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	// 每个线程负责将一部分数据写入共享内存
	shared[tx] = nums[id];
	__syncthreads();
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (tx % (stride * 2) == 0)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}
```

## 优化1：减少warp divergence

对于一个warp，里面的所有线程都执行同一条指令。对于if-else语句，如果warp中有的线程需要执行if中的语句，有的线程需要执行else中的语句，那么warp中的线程则会执行if-else中的所有语句，只是不满足条件的分支，不会得到计算结果。

上述实现中，如果线程i满足`tx % (stride * 2) == 0`，那么线程i+1则不会满足`tx % (stride * 2) == 0`，所以出现了分支。一般地，这两个相邻的线程会在一个warp中，所以warp中的所有线程都要执行if中的语句。

为了减少warp divergence，应该考虑将满足if条件的线程放在同一个warp，不满足if条件的线程放在另外的warp中。

![image-20211123114038441](https://gitee.com/huster_ning/image/raw/master//image/image-20211123114038441.png)
```cpp
/**
 * @brief 优化：减少分支
 * 
 */
__global__ void sum_2(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	shared[tx] = nums[id];
	__syncthreads();
	// 前半部分的数组和后半部分的数据进行sum操作
	for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
	{
		if (tx < stride)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}
```

## 优化2：解决bank冲突

## 优化3：解决Idle线程

上述代码在每轮迭代后就会有一半的线程闲置。特别有一半的线程只负责将全局内存读进共享内存，为了利用这些线程，我们可以让一个线程块处理`2 * BLOCK_SIZE`个数据的reduce操作。

```cpp
__global__ void sum_3(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	// 一个线程块负责2 * BLOCK_SIZE个数据的reduce操作
	shared[tx] = nums[id] + nums[id + blockDim.x];
	__syncthreads();
	// 前半部分的数组和后半部分的数据进行sum操作
	for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
	{
		if (tx < stride)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}
```

## 优化4：减少同步

迭代到后面stride小于32，则只有一个warp中的线程在工作，此时不需要使用`__syncthreads()`来进行线程块的同步。可以将最后几轮的循环给展开，避免同步操作。

> 一个warp中的32个线程其实是在一个SIMD单元上，这32个线程每次都是执行同一条指令，这天然地保持了同步状态，因而当s=32时，即只有一个SIMD单元在工作时，完全可以将__syncthreads()这条同步代码去掉

```cpp
__global__ void sum_4(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	// 一个线程块负责2 * BLOCK_SIZE个数据的reduce操作
	shared[tx] = nums[id] + nums[id + blockDim.x];
	__syncthreads();
	// 前半部分的数组和后半部分的数据进行sum操作
	for (int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tx < stride)
		{
			shared[tx] += shared[tx + stride];
		}
		__syncthreads();
	}
	// 最后一轮展开
	if(tx < 32){
		shared[tx] += shared[tx + 32];
		shared[tx] += shared[tx + 16];
		shared[tx] += shared[tx + 8];
		shared[tx] += shared[tx + 4];
		shared[tx] += shared[tx + 2];
		shared[tx] += shared[tx + 1];
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}
```

## 优化5：完全展开

对for循环进行完全展开，减少for循环的开销。

```cpp
/**
 * @brief 完全展开
 */
__global__ void sum_5(float *nums, float *res, unsigned int n)
{
	__shared__ float shared[BLOCK_SIZE];
	int tx = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	// 一个线程块负责2 * BLOCK_SIZE个数据的reduce操作
	shared[tx] = nums[id] + nums[id + blockDim.x];
	__syncthreads();
	int blockSize = blockDim.x;
	// 前半部分的数组和后半部分的数据进行sum操作
	// 循环完全展开
	if (blockSize >= 512 && tx < 256)
	{
		shared[tx] += shared[tx + 256];
	}
	__syncthreads();
	if (blockSize >= 256 && tx < 128)
	{
		shared[tx] += shared[tx + 128];
	}
	__syncthreads();
	if (blockSize >= 128 && tx < 64)
	{
		shared[tx] += shared[tx + 64];
	}
	__syncthreads();
	// 最后一轮展开
	if (tx < 32)
	{
		shared[tx] += shared[tx + 32];
		shared[tx] += shared[tx + 16];
		shared[tx] += shared[tx + 8];
		shared[tx] += shared[tx + 4];
		shared[tx] += shared[tx + 2];
		shared[tx] += shared[tx + 1];
	}
	if (tx == 0)
	{
		atomicAdd(res, shared[0]);
	}
}
```

## 优化6：reduce multiple elements per thread

上述代码一个线程处理两个元素的reduce，可以处理多个元素。
# 参考
1. [矩阵乘法优化](https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html)