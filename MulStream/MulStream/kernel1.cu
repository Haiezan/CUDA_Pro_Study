#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//(A+B)/2=C
#define N (1024)
#define FULL (N*20)

__global__ void kernel(int* a, int* b, int* c)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N)
	{
		c[idx] = (a[idx] + b[idx]) / 2;
	}
}

int run_with_1_stream()
{
	//初始化计时器事件
	cudaEvent_t start, stop;
	float elapsedTime;
	//声明流和Buffer指针
	cudaStream_t stream;
	int* host_a, * host_b, * host_c;
	int* dev_a, * dev_b, * dev_c;

	//创建计时器
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//初始化流
	cudaStreamCreate(&stream);

	//在GPU端申请存储空间
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	//在CPU端申请存储空间，用锁页内存
	cudaHostAlloc((void**)&host_a, FULL * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL * sizeof(int), cudaHostAllocDefault);

	//初始化A,B向量
	for (int i = 0; i < FULL; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	//开始计算
	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL; i += N)
	{
		//将数据从CPU锁页内存中传输给GPU显存
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		kernel << <N / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);
		//将计算结果从GPU显存中传输给CPU内存中
		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}
	cudaStreamSynchronize(stream);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("1 stream Time: %3.1f ms\n", elapsedTime);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaStreamDestroy(stream);
	return 0;
}

int run_with_2_stream()
{
	//初始化计时器事件
	cudaEvent_t start, stop;
	float elapsedTime;
	//声明流和Buffer指针
	cudaStream_t stream0;
	cudaStream_t stream1;
	int* host_a, * host_b, * host_c;
	int* dev_a0, * dev_b0, * dev_c0;
	int* dev_a1, * dev_b1, * dev_c1;

	//创建计时器
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//初始化流
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	//在GPU端申请存储空间
	cudaMalloc((void**)&dev_a0, N * sizeof(int));
	cudaMalloc((void**)&dev_b0, N * sizeof(int));
	cudaMalloc((void**)&dev_c0, N * sizeof(int));
	cudaMalloc((void**)&dev_a1, N * sizeof(int));
	cudaMalloc((void**)&dev_b1, N * sizeof(int));
	cudaMalloc((void**)&dev_c1, N * sizeof(int));
	//在CPU端申请存储空间，用锁页内存
	cudaHostAlloc((void**)&host_a, FULL * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL * sizeof(int), cudaHostAllocDefault);

	//初始化A,B向量
	for (int i = 0; i < FULL; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	//开始计算
	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL; i += N * 2)
	{
		//将数据从CPU锁页内存中传输给GPU显存
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		kernel << <N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
		kernel << <N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);
		//将计算结果从GPU显存中传输给CPU内存中
		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("2 stream Time: %3.1f ms\n", elapsedTime);

	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_a1);
	cudaFree(dev_b1);
	cudaFree(dev_c1);
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	return 0;
}


int main(void)
{
	//查询设备属性
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap)
	{
		printf("Device will not support overlap\n");
		return 0;
	}

	for (int i = 0; i < 1; i++)
	{
		//run_with_1_stream();
		run_with_2_stream();
	}

	return 0;
}