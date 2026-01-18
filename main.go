package main

import (
	"fmt"
	"os"
	"sync"
	"time"

	"cuda-test-golang/cuda"
)

func main() {
	const N = 1024 * 1024 * 128
	var wg sync.WaitGroup

	wg.Go(func() {
		// Preheat
		cuda.VectorsSum([]uint32{0, 1}, []uint32{2, 3})
	})

	a := make([]uint32, N)
	b := make([]uint32, N)

	for i := range a {
		a[i] = uint32(i)
		b[i] = uint32(i * 2)
	}

	wg.Wait()

	start := time.Now()
	if err := cuda.VectorsSum(a, b); err != cuda.CudaSuccess {
		os.Exit(int(err))
	}
	fmt.Printf("cuda.VectorsSum() elapsed time %0.3f ms\n", float64(time.Since(start).Microseconds())/1000.0)
}
