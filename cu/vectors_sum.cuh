#ifndef VECTORS_SUM_CUH
#define VECTORS_SUM_CUH

#include <stdint.h>

#define CUDA_EVENT_CREATE()  \
    float time;              \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop)

#define CUDA_EVENT_RECORD(start) cudaEventRecord(start, 0)

#define CUDA_EVENT_ELAPSED(time, start, stop) \
    cudaEventRecord(stop, 0);                 \
    cudaEventSynchronize(stop);               \
    cudaEventElapsedTime(&time, start, stop)

#ifdef CUDADLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT cudaError_t vectors_sum(uint32_t *a, uint32_t *b, uint32_t N);

#endif // VECTORS_SUM_CUH
