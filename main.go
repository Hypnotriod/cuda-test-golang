package main

import (
	"fmt"
	"log"
	"time"

	"cuda-test-golang/cuda"
)

func computeOnHost(a []uint32, b []uint32, c []uint32) {
	for i := range min(len(a), len(b), len(c)) {
		c[i] = a[i] + b[i]
	}
}

func verify(a []uint32, b []uint32) {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			log.Fatal("Verification failed!")
		}
	}
}

func main() {
	const N = 1024 * 1024 * 128

	pinnedMemA, err := cuda.NewPinnedMemory[uint32](N, cuda.CudaHostAllocMapped)
	if err != cuda.CudaSuccess {
		log.Fatal("Unable to allocate pinned memory. Error code:", err)
	}
	defer pinnedMemA.Free()
	pinnedMemB, err := cuda.NewPinnedMemory[uint32](N, cuda.CudaHostAllocMapped)
	if err != cuda.CudaSuccess {
		log.Fatal("Unable to allocate pinned memory. Error code:", err)
	}
	defer pinnedMemB.Free()

	fmt.Println("pinnedMemA size:", pinnedMemA.Size(), "length:", pinnedMemA.Length())

	a := pinnedMemA.Slice()
	b := pinnedMemB.Slice()
	c := make([]uint32, N)

	for i := range a {
		a[i] = uint32(i)
		b[i] = uint32(i * 2)
	}

	start := time.Now()
	computeOnHost(a, b, c)
	fmt.Printf("computeOnHost elapsed time: %0.3f ms\n", float64(time.Since(start).Microseconds())/1000.0)

	start = time.Now()
	if err := cuda.VectorAddUint32Mapped(a, b); err != cuda.CudaSuccess {
		log.Fatal("Unable to perform vector add. Error code:", err)
	}
	fmt.Printf("cuda.VectorAddUint32() elapsed time: %0.3f ms\n", float64(time.Since(start).Microseconds())/1000.0)

	verify(a, c)
	fmt.Println("Success")
}
