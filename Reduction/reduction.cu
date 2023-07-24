#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Time/Time_ns.h"

const int N = (1 << 20) - 3;
const int N_bytes = sizeof(float) * N;
const int threadsPerBlock = 512;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

// CPU串行计算函数
float CPUsum(float* x)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += x[i];
	}
	return sum;
}

// GPU单核串行计算函数
__global__ void GPUsum(const float* x, float* sum, const int* N)
{
	*sum = 0;
	for (int i = 0; i < *N; i++)
	{
		*sum += x[i];
	}
}

// kernel1
// 简单的树形结构并行，存在线程分歧
// Interleaved addressing with divergent branching
__global__ void kernel1(float* arr, float* out)
{
	__shared__ float s_data[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; // tid号线程负责的数组元素的位置
	
	// 将数据拷贝到共享内存
	if (i < N)
	{
		s_data[tid] = arr[i];
	}
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2)
	{
		// 偶数线程work，例如第一论循环中，只有0、2、4、6线程执行，1、3、5线程闲置，同一个warp内有一半线程没用上
		if (tid % (2 * s) == 0 && i + s < N) //线程分歧
		{
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	// 拷贝规约结果，注意是Block的和，而非最终结果
	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}

// kernel2
// 解决线程分歧，存在bank conflicts
__global__ void kernel2(float* arr, float* out)
{
	__shared__ float s_data[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; // tid号线程负责的数组元素的位置

	// 将数据拷贝到共享内存
	if (i < N)
	{
		s_data[tid] = arr[i];
	}
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2)
	{
		int index = tid * 2 * s;
		if ((index + s) < blockDim.x && (blockIdx.x * blockDim.x + index + s) < N)
		{
			s_data[index] += s_data[index + s];
		}
		__syncthreads();
	}

	// 拷贝规约结果，注意是Block的和，而非最终结果
	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}

// kernel3
// 连续地址，处理bank conflict问题
__global__ void kernel3(float* arr, float* out)
{
	__shared__ float s_data[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; // tid号线程负责的数组元素的位置

	// 将数据拷贝到共享内存
	if (i < N)
	{
		s_data[tid] = arr[i];
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) // 解决kernel2中share memory bank conflict
	{
		if (tid < s && i + s < N)
		{
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	// 拷贝规约结果，注意是Block的和，而非最终结果
	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}

// kernel4
// First add during global load
// share memory 拷贝时就进行一次加操作
__global__ void kernel4(float* arr, float* out)
{
	__shared__ float s_data[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x * 2; // tid号线程负责的数组元素的位置

	// 将数据拷贝到共享内存
	if (i < N)
	{
		s_data[tid] = arr[i] + arr[i + blockDim.x];
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) // 解决kernel2中share memory bank conflict
	{
		if (tid < s && i + s < N)
		{
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	// 拷贝规约结果，注意是Block的和，而非最终结果
	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}


int main(void)
{
	// 计时变量
	time_ns start, end;

	// 初始化数据
	float* x = new float[N]; //源数据
	float* y = new float[blocksPerGrid]; //并行计算中间数据
	float sum; //结果
	for(int i = 0; i < N; i++)
	{
		x[i] = 1;
	}
	for (int i = 0; i < blocksPerGrid; i++)
	{
		y[i] = 0;
	}
	//初始化GPU内存
	float* dx, * dy;
	float* dsum;
	int* dN;
	cudaMalloc((void**)&dx, N_bytes);
	cudaMalloc((void**)&dy, blocksPerGrid * sizeof(float));
	cudaMalloc((void**)&dsum, sizeof(float));
	cudaMalloc((void**)&dN, sizeof(int));
	cudaMemcpy(dx, x, N_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dN, &N, sizeof(int), cudaMemcpyHostToDevice);


	/*CPU串行*/
	start.clock();
	// 运行CPU串行函数
	sum = CPUsum(x);
	end.clock();

	// 输出CPU运行结果
	printf("Calculate sum by CPU: sum = %f\nElasped time: %fms\n", sum, end - start);
	printf("\n");

	/*GPU单核串行*/
	
	// 运行GPU串行函数
	start.clock();
	GPUsum <<<1, 1 >>> (dx, dsum, dN);

	// 拷贝计算结果到CPU
	cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU: sum = %f\nElasped time: %fms\n", sum, end - start);
	printf("\n");


	/*GPU并行 kernel1*/
	start.clock();
	sum = 0;
	kernel1 <<<blocksPerGrid, threadsPerBlock >>> (dx, dy);
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel1: sum = %f\nElasped time: %fms\n", sum, end - start);
	printf("\n");


	/*GPU并行 kernel2*/
	start.clock();
	sum = 0;
	kernel2 <<<blocksPerGrid, threadsPerBlock >>> (dx, dy);
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel2: sum = %f\nElasped time: %fms\n", sum, end - start);
	printf("\n");


	/*GPU并行 kernel3*/
	start.clock();
	sum = 0;
	kernel3 <<<blocksPerGrid, threadsPerBlock >>> (dx, dy);
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel3: sum = %f\nElasped time: %fms\n", sum, end - start);
	printf("\n");


	/*GPU并行 kernel4*/
	const int halfblocksPerGrid = blocksPerGrid / 2; //进程块数量减半
	start.clock();
	sum = 0;
	kernel4 <<<halfblocksPerGrid, threadsPerBlock >>> (dx, dy);
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, halfblocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < halfblocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel4: sum = %f\nElasped time: %fms\n", sum, end - start);
	printf("\n");



	cudaFree(dx);
	cudaFree(dsum);
	cudaFree(dN);
	free(x);
	free(y);
	return 0;
}