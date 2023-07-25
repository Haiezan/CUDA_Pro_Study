#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <time.h>
using namespace std;

__global__ void kernel()
{
	printf("blockIdx = %d, threadIdx = %d\n", blockIdx.x, threadIdx.x);
	//__syncthreads();
}

int main()
{
	int* dn;
	int n = 0;
	cudaMalloc((void**)&dn, sizeof(int));
	cudaMemcpy(dn, &n, sizeof(int), cudaMemcpyHostToDevice);

	kernel <<<1, 256 >>> ();
	cudaThreadSynchronize();
	system("pause");
	return 0;
}