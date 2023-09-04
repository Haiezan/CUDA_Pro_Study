
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../Time/Time_ns.h"
#include <cmath>

#define func pow

const int N = 1<<20;

int main()
{
    time_ns start, end;

    double* x_double = new double[N];
    double y_double = 2;
    double z_double;

    float* x_float = new float[N];
    float y_float = 2.f;
    float z_float;

    for (int i = 0; i < N; i++)
    {
        x_double[i] = i + 1;
        x_float[i] = i + 1;
    }

    // 1 double=func(double,double)
    start.clock();
    for (int i = 0; i < N; i++)
    {
        z_double = func(x_double[i], y_double);
    }
    end.clock();

    printf("Calculate double=func(double,double) = %f Elapsed time: %fms\n", z_double, (end - start));
    printf("\n");

    // 2 float=func(double,double)
    start.clock();
    for (int i = 0; i < N; i++)
    {
        z_float = func(x_double[i], y_double);
    }
    end.clock();

    printf("Calculate float=func(double,double) = %f Elapsed time: %fms\n", z_float, (end - start));
    printf("\n");

    // 3 double=func(double,float)
    start.clock();
    for (int i = 0; i < N; i++)
    {
        z_double = func(x_double[i], y_float);
    }
    end.clock();

    printf("Calculate double=func(double,float) = %f Elapsed time: %fms\n", z_double, (end - start));
    printf("\n");

    // 4 float=func(double,float)
    start.clock();
    for (int i = 0; i < N; i++)
    {
        z_float = func(x_double[i], y_float);
    }
    end.clock();

    printf("Calculate float=func(double,float) = %f Elapsed time: %fms\n", z_float, (end - start));
    printf("\n");

    // 5 double=func(float,float)
    start.clock();
    for (int i = 0; i < N; i++)
    {
        z_double = func(x_float[i], y_float);
    }
    end.clock();

    printf("Calculate double=func(float,float) = %f Elapsed time: %fms\n", z_double, (end - start));
    printf("\n");

    // 6 float=pow(float,float)
    start.clock();
    for (int i = 0; i < N; i++)
    {
        z_float = func(x_float[i], y_float);
    }
    end.clock();

    printf("Calculate float=func(float,float) = %f Elapsed time: %fms\n", z_float, (end - start));
    printf("\n");

    char a = getchar();

    return 0;
}
