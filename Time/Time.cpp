// Time.cpp : 实现毫秒级和纳秒级计时
//

#include <iostream>
#include <time.h>
#include <Windows.h>

float GetDuration(const timespec* end, const timespec* start)
{
    long d_sec = end->tv_sec - start->tv_sec;
    long d_nsec = end->tv_nsec - start->tv_nsec;
    return float(d_sec * 1e9 + d_nsec) / 1e6;
}

int main()
{
    clock_t start, end; // 毫秒级计时
    struct timespec start_ns, end_ns; // 纳秒级计时 C11标准库函数

    // 开始计时点
    start = clock();
    timespec_get(&start_ns, TIME_UTC);

    // 执行代码
    Sleep(1000); // ms

    // 结束计时点
    end = clock();
    timespec_get(&end_ns, TIME_UTC);

    // 输出时间
    printf("Elasped time: %lds\n", (end - start) / CLOCKS_PER_SEC);
    printf("Elasped time: %ldms\n", end - start);
    printf("Elasped time: %fms\n", GetDuration(&end_ns, &start_ns));
}
