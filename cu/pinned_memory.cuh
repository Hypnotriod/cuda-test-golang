#pragma once

cudaError_t malloc_host(uint8_t **ptr, size_t size, uint32_t flags)
{
    return cudaMallocHost(ptr, size, flags);
}

cudaError_t free_host(uint8_t **ptr)
{
    return cudaFreeHost(ptr);
}