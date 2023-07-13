#pragma once
#include <time.h>

struct time_ns: public timespec
{
    void clock()
    {
        timespec_get(this, TIME_UTC);
    }
};

float GetDuration(const timespec* end, const timespec* start)
{
    long d_sec = end->tv_sec - start->tv_sec;
    long d_nsec = end->tv_nsec - start->tv_nsec;
    return float(d_sec * 1e9 + d_nsec) / 1e6;
}


inline double operator-(const time_ns& end, const time_ns& start)
{
    long d_sec = end.tv_sec - start.tv_sec;
    long d_nsec = end.tv_nsec - start.tv_nsec;
    double f = (d_sec * 1e9 + d_nsec) / 1e6;
    return f;
}
