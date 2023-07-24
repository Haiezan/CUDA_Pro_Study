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
const int iters = 1000; // 核心循环计算次数，用于统计计算效率

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

// kernel5
// Unroll last warp
// 展开最后一个warp
__device__ void warpReduce(volatile float* s_data, int tid) // volatile貌似无影响
{
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}
__global__ void kernel5(float* arr, float* out)
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

	for (int s = blockDim.x / 2; s > 32; s >>= 1) // 当数据小于32的时候，对warp进行展开
	{
		if (tid < s && i + s < N)
		{
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	// 展开最后一个warp
	if (tid < 32)
	{
		//warpReduce(s_data, tid);
		s_data[tid] += s_data[tid + 32];
		s_data[tid] += s_data[tid + 16];
		s_data[tid] += s_data[tid + 8];
		s_data[tid] += s_data[tid + 4];
		s_data[tid] += s_data[tid + 2];
		s_data[tid] += s_data[tid + 1];
	}

	// 拷贝规约结果，注意是Block的和，而非最终结果
	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}

// kernel6
// Completely unroll
// 完全展开
template<unsigned int blockSize>
__device__ void warpReduce2(volatile float* s_data, int tid)
{
	if (blockSize >= 64) s_data[tid] += s_data[tid + 32];
	if (blockSize >= 32) s_data[tid] += s_data[tid + 16];
	if (blockSize >= 16) s_data[tid] += s_data[tid + 8];
	if (blockSize >= 8) s_data[tid] += s_data[tid + 4];
	if (blockSize >= 4) s_data[tid] += s_data[tid + 2];
	if (blockSize >= 2) s_data[tid] += s_data[tid + 1];
}
template<unsigned int blockSize>
__global__ void reduce(float* arr, float* out)
{
	__shared__ float s_data[threadsPerBlock];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x * 2; // tid号线程负责的数组元素的位置
	
	if (i < N)
	{
		s_data[tid] = arr[i] + arr[i + blockDim.x];
	}
	else
	{
		s_data[tid] = 0;
	}
	__syncthreads();

	if (blockSize >= 1024)
	{
		if (tid < 512)
		{
			s_data[tid] += s_data[tid + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			s_data[tid] += s_data[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			s_data[tid] += s_data[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			s_data[tid] += s_data[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		warpReduce2<blockSize>(s_data, tid);
	}

	if (tid == 0)
	{
		out[blockIdx.x] = s_data[0];
	}
}
void kernel6(float* arr, float* out) // 展开所有循环
{
	switch (threadsPerBlock)
	{
	case 1024:
		reduce<1024> <<<blocksPerGrid, threadsPerBlock >>> (arr, out); break;
	case 512:
		reduce<512> <<<blocksPerGrid, threadsPerBlock >>> (arr, out); break;
	case 256:
		reduce<256> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 128:
		reduce<128> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 64:
		reduce<64> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 32:
		reduce<32> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 16:
		reduce<16> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 8:
		reduce<8> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 4:
		reduce<4> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 2:
		reduce<2> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
	case 1:
		reduce<1> << <blocksPerGrid, threadsPerBlock >> > (arr, out); break;
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
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		sum = CPUsum(x);
	}
	end.clock();

	// 输出CPU运行结果
	printf("Calculate sum by CPU: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");

	/*GPU单核串行*/
	
	// 运行GPU串行函数
	start.clock();
	GPUsum <<<1, 1 >>> (dx, dsum, dN);

	// 拷贝计算结果到CPU
	cudaMemcpy(&sum, dsum, sizeof(int), cudaMemcpyDeviceToHost);
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU: sum = %f\nElapsed time: %fms\n", sum, end - start);
	printf("\n");


	/*GPU并行 kernel1*/
	start.clock();
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		kernel1 <<<blocksPerGrid, threadsPerBlock >>> (dx, dy);
	}
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel1: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");


	/*GPU并行 kernel2*/
	start.clock();
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		kernel2 <<<blocksPerGrid, threadsPerBlock >>> (dx, dy);
	}
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel2: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");


	/*GPU并行 kernel3*/
	start.clock();
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		kernel3 <<<blocksPerGrid, threadsPerBlock >>> (dx, dy);
	}
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel3: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");


	/*GPU并行 kernel4*/
	const int halfblocksPerGrid = blocksPerGrid / 2; //进程块数量减半
	start.clock();
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		kernel4 <<<halfblocksPerGrid, threadsPerBlock >>> (dx, dy);
	}
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, halfblocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < halfblocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel4: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");


	/*GPU并行 kernel5*/
	start.clock();
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		kernel5 <<<halfblocksPerGrid, threadsPerBlock >>> (dx, dy);
	}
	
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, halfblocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < halfblocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();
	 
	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel5: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");

	/*GPU并行 kernel6*/
	start.clock();
	for (int t = 0; t < iters; t++)
	{
		sum = 0;
		kernel6(dx, dy);
	}
	
	// 拷贝计算结果到CPU
	cudaMemcpy(y, dy, halfblocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < halfblocksPerGrid; i++)
	{
		sum += y[i];
	}
	end.clock();

	// 输出GPU串行计算结果
	printf("Calculate sum by GPU kernel6: sum = %f\nElapsed time: %fms\n", sum, (end - start) / iters);
	printf("\n");


	cudaFree(dx);
	cudaFree(dsum);
	cudaFree(dN);
	free(x);
	free(y);

	system("pause");
	return 0;
}