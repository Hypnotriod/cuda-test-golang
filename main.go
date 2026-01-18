package main

import (
	"C"
	"cuda-test-golang/cuda"
	"fmt"
	"os"
)

func main() {
	const N = 1024

	a := make([]uint32, N)
	b := make([]uint32, N)

	for i := range a {
		a[i] = uint32(i)
		b[i] = uint32(i * 2)
	}

	fmt.Println(a)

	if err := cuda.VectorsSum(a, b); err != cuda.CudaSuccess {
		os.Exit(int(err))
	}

	fmt.Println(a)
}
