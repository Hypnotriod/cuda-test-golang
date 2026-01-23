#pragma once

#include <stdint.h>

#ifdef CUDADLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT cudaError_t vectors_sum(uint32_t *a, uint32_t *b, uint32_t N);
extern "C" DLLEXPORT cudaError_t malloc_host(uint8_t **ptr, size_t size);
extern "C" DLLEXPORT cudaError_t free_host(uint8_t **ptr);