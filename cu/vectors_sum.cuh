#ifndef VECTORS_SUM_CUH
#define VECTORS_SUM_CUH

#include <stdio.h>
#include <stdint.h>

#ifdef CUDADLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT cudaError_t vectors_sum(uint32_t *a, uint32_t *b, uint32_t N);

#endif // VECTORS_SUM_CUH
