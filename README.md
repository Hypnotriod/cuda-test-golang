# cuda-test-golang

Test of how to build and call the `CUDA` library from `Golang`.

## Installation Windows
[Installation Guide Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
* Install the [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/) 
* Add the `Microsoft C++ Build Tools` to the `PATH` at `System Environment Variables`, ex.: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.35.32215\bin\Hostx64\x64`
* Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
* (Optionally) Install the [w64devkit](https://github.com/skeeto/w64devkit/releases/latest) for the `C/C++` build tools and standard unix utilities.

## CUDA
[CUDA Samples](https://github.com/NVIDIA/cuda-samples)
[Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
* Check CUDA installation
```sh
# Version of the Nvidia CUDA Compiler (nvcc)
nvcc --version
# NVIDIA System Management Interface
nvidia-smi
# Obtain the GPU Compute Capability
nvidia-smi --query-gpu=name,compute_cap
```

## Build and Run
```sh
# Build CUDA dll library and go executable
make build
# Run the executable
make run
```
