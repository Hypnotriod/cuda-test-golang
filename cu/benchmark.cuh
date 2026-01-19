#pragma once

class Benchmark
{
private:
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    Benchmark()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void record()
    {
        cudaEventRecord(start, 0);
    }

    float elapsed()
    {
        float time;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};
