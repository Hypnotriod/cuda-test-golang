#pragma once

#include <stdint.h>

#ifdef CUDADLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT cudaError_t malloc_host(uint8_t **ptr, size_t size, uint32_t flags);
extern "C" DLLEXPORT cudaError_t free_host(uint8_t **ptr);
extern "C" DLLEXPORT cudaError_t vector_add_uint32(uint32_t *a, uint32_t *b, uint32_t N);
extern "C" DLLEXPORT cudaError_t vector_add_uint32_mapped(uint32_t *a, uint32_t *b, uint32_t N);
