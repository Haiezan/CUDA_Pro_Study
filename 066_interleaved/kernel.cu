
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"
#include "assert.h"
#include "conio.h"
#include <stdlib.h>
#include <time.h>

#define CUDA_CALL(x){const cudaError_t a=(x);if(a!=cudaSuccess){printf("\nCUDA Error: %s (err_num=%d) \n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}

// Define the number of elements we'll use
#define NUM_ELEMENTS 4096
typedef unsigned int u32;

// Defien an interleaved type
// 16 bytes, 4 bytes per number
typedef struct
{
	u32 a;
	u32 b;
	u32 c;
	u32 d;
} INTERLEAVED_T;

// Define an array type base on the interleaved structure
typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

// Alternative - structure of arrays

typedef u32 ARRAY_NUMBER_T[NUM_ELEMENTS];

typedef struct
{
	ARRAY_NUMBER_T a;
	ARRAY_NUMBER_T b;
	ARRAY_NUMBER_T c;
	ARRAY_NUMBER_T d;
} NON_INTERLEAVED_T;

// In CPU
__host__ float add_test_non_interleaved_cpu(
	NON_INTERLEAVED_T* const dest_ptr,
	const NON_INTERLEAVED_T* const src_ptr,
	const u32 iter,
	const u32 num_elements
)
{
	clock_t start = clock();

	for (u32 tid = 0; tid < num_elements; tid++)
	{
		for (u32 i = 0; i < iter; i++)
		{
			dest_ptr->a[tid] += src_ptr->a[tid];
			dest_ptr->b[tid] += src_ptr->b[tid];
			dest_ptr->c[tid] += src_ptr->c[tid];
			dest_ptr->d[tid] += src_ptr->d[tid];
		}
	}

	clock_t stop = clock();

	float delta = stop - start;
	return delta;
}

__host__ float add_test_interleaved_cpu(
	INTERLEAVED_T* const dest_ptr,
	const INTERLEAVED_T* const src_ptr,
	const u32 iter,
	const u32 num_elements
)
{
	clock_t start = clock();

	for (u32 tid = 0; tid < num_elements; tid++)
	{
		for (u32 i = 0; i < iter; i++)
		{
			dest_ptr[tid].a += src_ptr[tid].a;
			dest_ptr[tid].b += src_ptr[tid].b;
			dest_ptr[tid].c += src_ptr[tid].c;
			dest_ptr[tid].d += src_ptr[tid].d;
		}
	}

	clock_t stop = clock();

	float delta = stop - start;
	return delta;
}

// In GPU
__global__ void add_kernel_non_interleaved(
	NON_INTERLEAVED_T* const host_dest_ptr,
	const NON_INTERLEAVED_T* const host_src_ptr,
	const u32 iter,
	const u32 num_elements
)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < num_elements)
	{
		for (u32 i = 0; i < iter; i++)
		{
			host_dest_ptr->a[tid] += host_src_ptr->a[tid];
			host_dest_ptr->b[tid] += host_src_ptr->b[tid];
			host_dest_ptr->c[tid] += host_src_ptr->c[tid];
			host_dest_ptr->d[tid] += host_src_ptr->d[tid];
		}
	}
}

__global__ void add_kernel_interleaved(
	INTERLEAVED_T* const host_dest_ptr,
	const INTERLEAVED_T* const host_src_ptr,
	const u32 iter,
	const u32 num_elements
)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < num_elements)
	{
		for (u32 i = 0; i < iter; i++)
		{
			host_dest_ptr[tid].a += host_src_ptr[tid].a;
			host_dest_ptr[tid].b += host_src_ptr[tid].b;
			host_dest_ptr[tid].c += host_src_ptr[tid].c;
			host_dest_ptr[tid].d += host_src_ptr[tid].d;
		}
	}
}

// Call function
__host__ float add_test_interleaved(
	INTERLEAVED_T* const host_dest_ptr,
	const INTERLEAVED_T* const host_src_ptr,
	const u32 iter,
	const u32 num_elements
)
{
	// Set launch params
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads - 1)) / num_threads;

	// Allocate memeory ont the device
	const size_t num_bytes = sizeof(INTERLEAVED_T) * num_elements;
	INTERLEAVED_T* device_dest_ptr;
	INTERLEAVED_T* device_src_ptr;

	CUDA_CALL(cudaMalloc((void**)&device_src_ptr, num_bytes));
	CUDA_CALL(cudaMalloc((void**)&device_dest_ptr, num_bytes));

	// Create a start and stop event for timing
	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start, 0);
	cudaEventCreate(&kernel_stop, 0);

	// Create a non zero stream
	cudaStream_t test_stream;
	CUDA_CALL(cudaStreamCreate(&test_stream));

	// Copy src data to GPU
	CUDA_CALL(cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes, cudaMemcpyHostToDevice));

	// Push start event ahead of kernel call
	CUDA_CALL(cudaEventRecord(kernel_start, 0));

	// Call the GPU kernel
	add_kernel_interleaved << <num_blocks, num_threads >> > (device_dest_ptr, device_src_ptr, iter, num_elements);

	// Push stop event after kernel call
	CUDA_CALL(cudaEventRecord(kernel_stop, 0));

	// Wait for stop event
	CUDA_CALL(cudaEventSynchronize(kernel_stop));

	// Copy dest data to CPU
	CUDA_CALL(cudaMemcpy(host_dest_ptr, device_dest_ptr, num_bytes, cudaMemcpyDeviceToHost));

	// Get delta between start and stop
	// i.e. the kernel execution time
	float delta = 0.f;
	CUDA_CALL(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));

	// Clean up
	CUDA_CALL(cudaFree(device_dest_ptr));
	CUDA_CALL(cudaFree(device_src_ptr));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(test_stream));

	return delta;
}
__host__ float add_test_non_interleaved(
	NON_INTERLEAVED_T* const host_dest_ptr,
	const NON_INTERLEAVED_T* const host_src_ptr,
	const u32 iter,
	const u32 num_elements
)
{
	// Set launch params
	const u32 num_threads = 256;
	const u32 num_blocks = (num_elements + (num_threads - 1)) / num_threads;

	// Allocate memeory ont the device
	const size_t num_bytes = sizeof(NON_INTERLEAVED_T);
	NON_INTERLEAVED_T* device_dest_ptr;
	NON_INTERLEAVED_T* device_src_ptr;

	CUDA_CALL(cudaMalloc((void**)&device_src_ptr, num_bytes));
	CUDA_CALL(cudaMalloc((void**)&device_dest_ptr, num_bytes));

	// Create a start and stop event for timing
	cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start, 0);
	cudaEventCreate(&kernel_stop, 0);

	// Create a non zero stream
	cudaStream_t test_stream;
	CUDA_CALL(cudaStreamCreate(&test_stream));

	// Copy src data to GPU
	CUDA_CALL(cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes, cudaMemcpyHostToDevice));

	// Push start event ahead of kernel call
	CUDA_CALL(cudaEventRecord(kernel_start, 0));

	// Call the GPU kernel
	add_kernel_non_interleaved << <num_blocks, num_threads >> > (device_dest_ptr, device_src_ptr, iter, num_elements);

	// Push stop event after kernel call
	CUDA_CALL(cudaEventRecord(kernel_stop, 0));

	// Wait for stop event
	CUDA_CALL(cudaEventSynchronize(kernel_stop));

	// Copy dest data to CPU
	CUDA_CALL(cudaMemcpy(host_dest_ptr, device_dest_ptr, num_bytes, cudaMemcpyDeviceToHost));

	// Get delta between start and stop
	// i.e. the kernel execution time
	float delta = 0.f;
	CUDA_CALL(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));

	// Clean up
	CUDA_CALL(cudaFree(device_dest_ptr));
	CUDA_CALL(cudaFree(device_src_ptr));
	CUDA_CALL(cudaEventDestroy(kernel_start));
	CUDA_CALL(cudaEventDestroy(kernel_stop));
	CUDA_CALL(cudaStreamDestroy(test_stream));

	return delta;
}

int main()
{
	INTERLEAVED_ARRAY_T host_dest_ptr;
	INTERLEAVED_ARRAY_T host_src_ptr;

	NON_INTERLEAVED_T host_non_dest_ptr;
	NON_INTERLEAVED_T host_non_src_ptr;

	for (u32 i = 0; i < NUM_ELEMENTS; i++)
	{
		host_dest_ptr[i].a = 0;
		host_dest_ptr[i].b = 0;
		host_dest_ptr[i].c = 0;
		host_dest_ptr[i].d = 0;

		host_src_ptr[i].a = 1;
		host_src_ptr[i].b = 1;
		host_src_ptr[i].c = 1;
		host_src_ptr[i].d = 1;

		host_non_dest_ptr.a[i] = 0;
		host_non_dest_ptr.b[i] = 0;
		host_non_dest_ptr.c[i] = 0;
		host_non_dest_ptr.d[i] = 0;

		host_non_src_ptr.a[i] = 1;
		host_non_src_ptr.b[i] = 1;
		host_non_src_ptr.c[i] = 1;
		host_non_src_ptr.d[i] = 1;
	}

	int iters = 10000;

	int max_device_num;
	CUDA_CALL(cudaGetDeviceCount(&max_device_num));
	for (int device_num = 0; device_num < max_device_num; device_num++)
	{
		CUDA_CALL(cudaSetDevice(device_num));

		struct cudaDeviceProp device_prop;
		char device_prefix[261];
		CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));
		sprintf(device_prefix, "ID:%d %s:", 0, device_prop.name);
		float time_gpu;

		time_gpu = add_test_interleaved(host_dest_ptr, host_src_ptr, iters, NUM_ELEMENTS);
		printf("\n%sInterleaved time : %fms", device_prefix, time_gpu);

		time_gpu = add_test_non_interleaved(&host_non_dest_ptr, &host_non_src_ptr, iters, NUM_ELEMENTS);
		printf("\n%sNon Interleaved time : %fms", device_prefix, time_gpu);
	}

	float time_cpu;
	time_cpu = add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, iters, NUM_ELEMENTS);
	printf("\nCPU: Interleaved time : %fms", time_cpu);
	time_cpu = add_test_non_interleaved_cpu(&host_non_dest_ptr, &host_non_src_ptr, iters, NUM_ELEMENTS);
	printf("\nCPU: Non Interleaved time : %fms", time_cpu);

	return 0;
}