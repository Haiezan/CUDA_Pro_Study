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

int main(void)
{
	//��ѯ�豸����
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if (!prop.deviceOverlap)
	{
		printf("Device will not support overlap\n");
		return 0;
	}

	//��ʼ����ʱ���¼�
	cudaEvent_t start, stop;
	float elapsedTime;
	//��������Bufferָ��
	cudaStream_t stream0;
	cudaStream_t stream1;
	int* host_a, * host_b, * host_c;
	int* dev_a0, * dev_b0, * dev_c0;
	int* dev_a1, * dev_b1, * dev_c1;

	//������ʱ��
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//��ʼ����
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	//��GPU������洢�ռ�
	cudaMalloc((void**)&dev_a0, N * sizeof(int));
	cudaMalloc((void**)&dev_b0, N * sizeof(int));
	cudaMalloc((void**)&dev_c0, N * sizeof(int));
	cudaMalloc((void**)&dev_a1, N * sizeof(int));
	cudaMalloc((void**)&dev_b1, N * sizeof(int));
	cudaMalloc((void**)&dev_c1, N * sizeof(int));
	//��CPU������洢�ռ䣬����ҳ�ڴ�
	cudaHostAlloc((void**)&host_a, FULL * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL * sizeof(int), cudaHostAllocDefault);

	//��ʼ��A,B����
	for (int i = 0; i < FULL; i++)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	//��ʼ����
	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL; i += N*2)
	{
		//�����ݴ�CPU��ҳ�ڴ��д����GPU�Դ�
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		kernel << <N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);
		kernel << <N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);
		//����������GPU�Դ��д����CPU�ڴ���
		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time: %3.1f ms\n", elapsedTime);

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