
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

const int N = (1 << 10) - 3;
const int N_bytes = sizeof(float) * N * N;
const int threadsPerBlock = 512;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
const int iters = 10; // 核心循环计算次数，用于统计计算效率

__global__ void transpose1(const float* A, float* B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}

__global__ void transpose2(const float* A, float* B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

int main()
{
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* hA = new float[N * N];
    float* hB = new float[N * N];
    for (int i = 0; i < N * N; i++)
    {
        hA[i] = 1.f;
        hB[i] = 1.f;
    }

    float* dA, * dB;
    cudaMalloc((void**)&dA, N_bytes);
    cudaMalloc((void**)&dB, N_bytes);

    cudaMemcpy(dA, hA, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N_bytes, cudaMemcpyHostToDevice);

    printf("transpose1 consume time: \n");
    for (int i = 0; i < iters; i++)
    {
        cudaEventRecord(start, 0);
        transpose1 << <blocksPerGrid, threadsPerBlock >> > (dA, dB, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("Time = %f ms\n", elapsed);
    }

    printf("transpose2 consume time: \n");
    for (int i = 0; i < iters; i++)
    {
        cudaEventRecord(start);
        transpose2 << <blocksPerGrid, threadsPerBlock >> > (dA, dB, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("Time = %f ms\n", elapsed);
    }

    cudaFree(dA);
    cudaFree(dB);
    free(hA);
    free(hB);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}