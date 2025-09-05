#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    }

#define CUBLAS_CHECK(call) \
    { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error" << std::endl; \
            exit(1); \
        } \
    }

int main() {
    const int N = 100000000; // 1e8 数据
    std::vector<float> h_data(N);

    // 随机数生成
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (int i = 0; i < N; i++) {
        h_data[i] = dist(rng);
    }

    // ------------------------------
    // 1. CPU 单线程
    // ------------------------------
    auto t1 = std::chrono::high_resolution_clock::now();

    float maxVal_cpu = -1e30f;
    int maxIdx_cpu = -1;
    for (int i = 0; i < N; i++) {
        if (h_data[i] > maxVal_cpu) {
            maxVal_cpu = h_data[i];
            maxIdx_cpu = i;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "[CPU 单线程] 最大值: " << maxVal_cpu
        << " 位置: " << maxIdx_cpu
        << " 时间: " << cpu_time << " 秒\n";

    // ------------------------------
    // 2. OpenMP 并行
    // ------------------------------
    double t3 = omp_get_wtime();

    float maxVal_omp = -1e30f;
    int maxIdx_omp = -1;

#pragma omp parallel
    {
        //int tid = omp_get_thread_num();     // 线程 ID
        //int nthreads = omp_get_num_threads(); // 总线程数

        float localMax = -1e30f;
        int localIdx = -1;

#pragma omp for nowait
        for (int i = 0; i < N; i++) {
            if (h_data[i] > localMax) {
                localMax = h_data[i];
                localIdx = i;
            }
        }

#pragma omp critical
        {
            //std::cout << "线程 " << tid << " / " << nthreads << " 正在运行\n";
            if (localMax > maxVal_omp) {
                maxVal_omp = localMax;
                maxIdx_omp = localIdx;
            }
        }
    }

    double t4 = omp_get_wtime();

    std::cout << "[OpenMP 并行] 最大值: " << maxVal_omp
        << " 位置: " << maxIdx_omp
        << " 时间: " << (t4 - t3) << " 秒\n";

    // ------------------------------
    // 3. GPU cuBLAS
    // ------------------------------
    float* d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int maxIdx_gpu = -1;
    auto t5 = std::chrono::high_resolution_clock::now();

    // 注意：cublasIsamax 返回 1-based 索引
    CUBLAS_CHECK(cublasIsamax(handle, N, d_data, 1, &maxIdx_gpu));

    auto t6 = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(t6 - t5).count();

    float maxVal_gpu = h_data[maxIdx_gpu - 1];

    std::cout << "[GPU cuBLAS] 最大值: " << maxVal_gpu
        << " 位置: " << (maxIdx_gpu - 1)
        << " 时间: " << gpu_time << " 秒\n";

    // ------------------------------
    // 清理
    // ------------------------------
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
