#include "vectors_sum.cuh"

__global__ static void kernel(uint32_t *a, uint32_t *b, size_t N)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        a[idx] += b[idx];
    }
}

cudaError_t vectors_sum(uint32_t *h_a, uint32_t *h_b, uint32_t N)
{
    size_t N_bytes = N * sizeof(uint32_t);
    size_t blockSize = 1024;
    size_t gridSize = (N + blockSize - 1) / blockSize;

    uint32_t *d_a;
    uint32_t *d_b;

    cudaError_t err;

    err = cudaMalloc(&d_a, N_bytes);
    if (err != cudaError::cudaSuccess)
        return err;
    err = cudaMalloc(&d_b, N_bytes);
    if (err != cudaError::cudaSuccess)
        return err;

    err = cudaMemcpy(d_a, h_a, N_bytes, cudaMemcpyHostToDevice);
    if (err != cudaError::cudaSuccess)
        return err;
    err = cudaMemcpy(d_b, h_b, N_bytes, cudaMemcpyHostToDevice);
    if (err != cudaError::cudaSuccess)
        return err;

    kernel<<<gridSize, blockSize>>>(d_a, d_b, N);
    err = cudaDeviceSynchronize();
    if (err != cudaError::cudaSuccess)
        return err;

    err = cudaMemcpy(h_a, d_a, N_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaError::cudaSuccess)
        return err;

    return cudaError::cudaSuccess;
}
