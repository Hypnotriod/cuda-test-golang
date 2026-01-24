#pragma once

cudaError_t malloc_host(uint8_t **ptr, size_t size)
{
    return cudaMallocHost(ptr, size);
}

cudaError_t free_host(uint8_t **ptr)
{
    return cudaFreeHost(ptr);
}