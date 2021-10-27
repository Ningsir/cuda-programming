# 基础

`MPI_Comm_size` 会返回 communicator 的大小，也就是 communicator 中可用的进程数量。在我们的例子中，`MPI_COMM_WORLD`（这个 communicator 是 MPI 帮我们生成的）这个变量**包含了当前 MPI 任务中所有的进程**，因此在我们的代码里的这个调用会返回所有的可用的进程数目。

```cpp
MPI_Comm_size(
    MPI_Comm communicator,
    int* size)
```

`MPI_Comm_rank` 这个函数会返回 communicator 中当前进程的 rank。 communicator 中每个进程会以此得到一个从0开始递增的数字作为 rank 值。rank 值主要是用来指定发送或者接受信息时对应的进程。

```cpp
MPI_Comm_rank(
    MPI_Comm communicator,
    int* rank)
```

## 通讯器

> `MPI_COMM_WORLD`就是一个通讯器，包含所有的进程

创建新的通讯器，以便一次与原始进程组的**子集进行通讯**。

```cpp
MPI_Comm_split(
	MPI_Comm comm,
	int color,
	int key,
	MPI_Comm* newcomm)
```

顾名思义，`MPI_Comm_split` 通过基于输入值 `color` 和 `key` 将通讯器“拆分”为一组子通讯器来创建新的通讯器。 在这里需要注意的是，原始的通讯器并没有消失，但是在每个进程中都会创建一个新的通讯器。 第一个参数 `comm` 是通讯器，它将用作新通讯器的基础。 这可能是 `MPI_COMM_WORLD`，但也可能是其他任何通讯器。 第二个参数 `color` 确定每个进程将属于哪个新的通讯器。 为 `color` 传递相同值的所有进程都分配给同一通讯器。 如果 `color` 为 `MPI_UNDEFINED`，则该进程将不包含在任何新的通讯器中。 第三个参数 `key` 确定每个新通讯器中的顺序（秩）。 传递 `key` 最小值的进程将为 0，下一个最小值将为 1，依此类推。 如果存在平局，则在原始通讯器中秩较低的进程将是第一位。 最后一个参数 `newcomm` 是 MPI 如何将新的通讯器返回给用户。

# 点对点通信

`MPI_Send`和`MPI_Recv`用于点对点通信，两者需要配合使用，它们会一直阻塞（同步通信），直到发送或者接收到消息。

如下代码，进程号不为0的进程会一直处于阻塞状态：

```cpp
// 点对点通信
	// send message
	if (world_rank == 0)
	{
		cout << "no messages" << endl;
	}
	// receive message，没有接收到消息就会一直阻塞
	else
	{
		int msg;
		MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cout << "message: " << msg << endl;
	}
```

> 如何确定没有消息发送

## MPI_Send和MPI_Recv

> 同步



## MPI_Isend和MPI_Irecv

> 异步

## 动态数据



# 集体通信

集体通信在进程间需要同步。

每一个你调用的集体通信方法都是同步的。也就是说，如果你没法让所有进程都完成 `MPI_Barrier`，那么你也没法完成任何集体调用。

```cpp
// 用于进程间同步
MPI_Barrier(MPI_Comm communicator)
```

树形广播算法

> 为什么不使用多线程进行广播？？

假设每个进程都只有**一个**「输出/输入」网络连接。



> MPI中集体通信对应CUDA编程？？

## MPI_Bcast

一个广播发生的时候，一个进程会把**同样一份数据**传递给一个 communicator 里的所有其他进程。广播的主要用途之一是把用户输入传递给一个分布式程序，或者把一些配置参数传递给所有的进程。

```cpp
MPI_Bcast(
    void* data,
    int count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm communicator)
```

`MPI_Bcast` 的实现使用了一个类似的**树形广播算法**来获得比较好的网络利用率。

![image-20211024144716664](https://gitee.com/huster_ning/image/raw/master//image/image-20211024144716664.png)

## MPI_Scatter

> 一对多

跟进程将数据发送到communicator里面的所有进程，但是`MPI_Scatter` 给每个进程发送的是*一个数组的一部分数据*。接收者和发送者都是调用此函数。

```cpp
MPI_Scatter(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_datatype,
    int root,
    MPI_Comm communicator)
```

> 要是发送的数据长度不一样长呢？？

## MPI_Gather

> 多对一

`MPI_Gather` 从多进程里面收集数据到一个进程。

```cpp
MPI_Gather(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_datatype,
    int root,
    MPI_Comm communicator)
```

*recv_count*参数是从*每个进程*接收到的数据数量，而不是所有进程的数据总量之和

## MPI_Allgather

> 多对多通信

每个进程将数据发送到其他进程。跟`MPI_Gather`一样，每个进程上的元素是根据他们的秩为顺序被收集起来的

![image-20211024150032406](https://gitee.com/huster_ning/image/raw/master//image/image-20211024150032406.png)

## MPI_Reduce

`MPI_Reduce` 在每个进程上获取一个输入元素数组，并将输出元素数组返回给根进程。

数据归约包括通过函数将一组数字归约为较小的一组数字。 MPI 定义的归约操作包括：

- `MPI_MAX` - 返回最大元素。
- `MPI_MIN` - 返回最小元素。
- `MPI_SUM` - 对元素求和。
- `MPI_PROD` - 将所有元素相乘。
- `MPI_LAND` - 对元素执行逻辑*与*运算。
- `MPI_LOR` - 对元素执行逻辑*或*运算。
- `MPI_BAND` - 对元素的各个位按位*与*执行。
- `MPI_BOR` - 对元素的位执行按位*或*运算。
- `MPI_MAXLOC` - 返回最大值和所在的进程的秩。
- `MPI_MINLOC` - 返回最小值和所在的进程的秩。

![image-20211024151758508](https://gitee.com/huster_ning/image/raw/master//image/image-20211024151758508.png)

注意：每个进程的输入数组长度为2，规约结果的长度也为2。

## MPI_Allreduce

许多并行程序中，需要在所有进程而不是仅仅在根进程中访问归约的结果。 以与 `MPI_Gather` 相似的补充方式，`MPI_Allreduce` 将归约值并将结果分配给所有进程

> 等效于：MPI_Reduce --> MPI_Bcast