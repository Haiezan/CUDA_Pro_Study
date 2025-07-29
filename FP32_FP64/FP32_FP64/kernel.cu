
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdlib>

// 定义测试向量长度
const size_t N = 1 << 24; // 16M元素
const int BLOCK_SIZE = 256;
const int iter = 30;

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// 单精度点积核函数（使用共享内存规约）
__global__ void dotProductFP32(const float* a, const float* b, float* result, size_t n) {
    __shared__ float shared_mem[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    // 每个线程负责多个元素
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    // 存入共享内存
    shared_mem[tid] = sum;
    __syncthreads();

    // block 内规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }

    // block 0线程0使用原子操作累加到 result
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

// 双精度点积核函数（使用共享内存规约）
__global__ void dotProductFP64(const double* a, const double* b, double* result, size_t n) {
    __shared__ double shared_mem[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    // 每个线程负责多个元素
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    // 存入共享内存
    shared_mem[tid] = sum;
    __syncthreads();

    // block 内规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }

    // block 0线程0使用原子操作累加到 result
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

// CPU参考计算（双精度）
double cpuDotProduct(const double* a, const double* b, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    // 分配主机内存
    float* h_a_fp32 = new float[N];
    float* h_b_fp32 = new float[N];
    double* h_a_fp64 = new double[N];
    double* h_b_fp64 = new double[N];

    // 初始化数据
    for (size_t i = 0; i < N; ++i) {
        float val = (i % 100) * 0.01f;
        h_a_fp32[i] = val;
        h_b_fp32[i] = 1.0f / (val + 0.1f);
        h_a_fp64[i] = static_cast<double>(val);
        h_b_fp64[i] = 1.0 / (static_cast<double>(val) + 0.1);
    }

    // CPU参考计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = cpuDotProduct(h_a_fp64, h_b_fp64, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    // 分配设备内存
    float* d_a_fp32, * d_b_fp32, * d_result_fp32;
    double* d_a_fp64, * d_b_fp64, * d_result_fp64;

    CHECK_CUDA_ERROR(cudaMalloc(&d_a_fp32, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_fp32, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result_fp32, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_a_fp64, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_fp64, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result_fp64, sizeof(double)));

    // 拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_a_fp32, h_a_fp32, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b_fp32, h_b_fp32, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a_fp64, h_a_fp64, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b_fp64, h_b_fp64, N * sizeof(double), cudaMemcpyHostToDevice));

    // 计算网格和块大小
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x);

    // 单精度测试
    CHECK_CUDA_ERROR(cudaMemset(d_result_fp32, 0, sizeof(float)));
    cudaDeviceSynchronize();
    auto fp32_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
    {
        CHECK_CUDA_ERROR(cudaMemset(d_result_fp32, 0, sizeof(float)));
        cudaDeviceSynchronize();
        dotProductFP32 << <grid, block >> > (d_a_fp32, d_b_fp32, d_result_fp32, N);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto fp32_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fp32_duration = (fp32_end - fp32_start) / iter;

    float h_result_fp32;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result_fp32, d_result_fp32, sizeof(float), cudaMemcpyDeviceToHost));

    // 双精度测试
    CHECK_CUDA_ERROR(cudaMemset(d_result_fp64, 0, sizeof(double)));
    cudaDeviceSynchronize();
    auto fp64_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
    {
        CHECK_CUDA_ERROR(cudaMemset(d_result_fp64, 0, sizeof(double)));
        cudaDeviceSynchronize();
        dotProductFP64 << <grid, block >> > (d_a_fp64, d_b_fp64, d_result_fp64, N);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto fp64_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fp64_duration = (fp64_end - fp64_start) / iter;

    double h_result_fp64;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result_fp64, d_result_fp64, sizeof(double), cudaMemcpyDeviceToHost));

    // 计算误差
    double fp32_error = fabs(static_cast<double>(h_result_fp32) - cpu_result) / cpu_result;
    double fp64_error = fabs(h_result_fp64 - cpu_result) / cpu_result;

    // 输出结果
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "CPU reference result (FP64): " << cpu_result << std::endl;
    std::cout << "GPU FP32 result: " << h_result_fp32 << std::endl;
    std::cout << "GPU FP64 result: " << h_result_fp64 << std::endl;
    std::cout << "\nError comparison:" << std::endl;
    std::cout << "FP32 relative error: " << fp32_error * 100.0 << "%" << std::endl;
    std::cout << "FP64 relative error: " << fp64_error * 100.0 << "%" << std::endl;
    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;
    std::cout << "FP32 GPU time: " << fp32_duration.count() << " seconds" << std::endl;
    std::cout << "FP64 GPU time: " << fp64_duration.count() << " seconds" << std::endl;
    std::cout << "FP32 speedup over CPU: " << cpu_duration.count() / fp32_duration.count() << "x" << std::endl;
    std::cout << "FP64 speedup over CPU: " << cpu_duration.count() / fp64_duration.count() << "x" << std::endl;
    std::cout << "FP32 vs FP64 speed ratio: " << fp64_duration.count() / fp32_duration.count() << "x" << std::endl;

    // 释放内存
    delete[] h_a_fp32;
    delete[] h_b_fp32;
    delete[] h_a_fp64;
    delete[] h_b_fp64;

    CHECK_CUDA_ERROR(cudaFree(d_a_fp32));
    CHECK_CUDA_ERROR(cudaFree(d_b_fp32));
    CHECK_CUDA_ERROR(cudaFree(d_result_fp32));
    CHECK_CUDA_ERROR(cudaFree(d_a_fp64));
    CHECK_CUDA_ERROR(cudaFree(d_b_fp64));
    CHECK_CUDA_ERROR(cudaFree(d_result_fp64));

    return 0;
}