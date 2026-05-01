#pragma once

#include <stdint.h>

#if defined(_MSC_VER)
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT __attribute__((visibility("default")))
#else
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#ifdef CUDADLL_EXPORTS
    #define DLLEXPORT EXPORT
#else
    #define DLLEXPORT IMPORT
#endif

extern "C" DLLEXPORT cudaError_t malloc_host(uint8_t **ptr, size_t size, uint32_t flags);
extern "C" DLLEXPORT cudaError_t free_host(uint8_t **ptr);
extern "C" DLLEXPORT cudaError_t vector_add_uint32(uint32_t *a, uint32_t *b, uint32_t N);
extern "C" DLLEXPORT cudaError_t vector_add_uint32_mapped(uint32_t *a, uint32_t *b, uint32_t N);
