#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

// GPU单核穿行计算函数
__global__ void GPUsum(const float* x, float* sum, const int* N)
{
	*sum = 0;
	for (int i = 0; i < *N; i++)
	{
		*sum += x[i];
	}
}

// 简单的树形结构并行
__global__ void kernel1(float* arr, float* out)
{
	__shared__ float s_data[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; // tid号线程负责的数组元素的位置
	if (i < N)
	{
		s_data[tid] = arr[i];
	}
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0 && i + s < N)
		{
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}

}



int main(void)
{
	// 计时变量
	clock_t start, end;

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
	start = clock();
	// 运行CPU串行函数
	sum = CPUsum(x);
	end = clock();

	// 输出CPU运行结果
	printf("Calculate sum by CPU: sum = %f\nElasped time: %ldms\n", sum, (end - start));
	printf("\n");

	/*GPU单核串行*/
	
	// 运行GPU串行函数
	start = clock();
	GPUsum << <1, 1 >> > (dx, dsum, dN);

	// 拷贝计算结果到CPU
	cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU: sum = %f\nElasped time: %ldms\n", sum, (end - start));
	printf("\n");


	/*GPU并行 kernel1*/
	start = clock();
	sum = 0;
	kernel1 << <blocksPerGrid, threadsPerBlock >> > (dx, dy);
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end = clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel1: sum = %f\nElasped time: %ldms\n", sum, (end - start));
	printf("\n");



	cudaFree(dx);
	cudaFree(dsum);
	cudaFree(dN);
	free(x);
	free(y);
	return 0;
}