#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// CPU串行计算函数
float CPUsum(const float* x, const int N)
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

#define ARRAY_SIZE 1<<20
#define ARRAY_SIZE_IN_BYTES (sizeof(float) * (ARRAY_SIZE))

int main(void)
{
	// 计时变量
	clock_t start, end;

	// 初始化数据
	int N = ARRAY_SIZE;
	float* x = new float[ARRAY_SIZE];
	float sum;
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		x[i] = 1;
	}
	//初始化GPU内存
	float* dx;
	float* dsum;
	int* dN;
	cudaMalloc((void**)&dx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&dsum, sizeof(float));
	cudaMalloc((void**)&dN, sizeof(int));
	cudaMemcpy(dx, x, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(dN, &N, sizeof(int), cudaMemcpyHostToDevice);


	/*CPU串行*/
	start = clock();
	// 运行CPU串行函数
	sum = CPUsum(x, ARRAY_SIZE);
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





	cudaFree(dx);
	cudaFree(dsum);
	cudaFree(dN);
	return 0;
}