package cuda

/*
#include <stdint.h>
#include "cuda_types.h"

cudaError_t malloc_host(uint8_t **ptr, size_t size);
cudaError_t free_host(uint8_t **ptr);
*/
import "C"
import (
	"unsafe"
)

type PinnedMemory[T any] struct {
	ptr    *C.uint8_t
	length uint
	size   uint
	valid  bool
}

func NewPinnedMemory[T any](length uint) (*PinnedMemory[T], CudaError) {
	var v T
	m := &PinnedMemory[T]{
		length: length,
		size:   uint(unsafe.Sizeof(v)),
	}
	err := C.malloc_host(&m.ptr, C.size_t(m.length*m.size))
	if CudaError(err) == CudaSuccess {
		m.valid = true
	}
	return m, CudaError(err)
}

func (m *PinnedMemory[T]) Slice() []T {
	if !m.valid {
		var slice []T
		return slice
	}
	var ptr *T = (*T)(unsafe.Pointer(m.ptr))
	return unsafe.Slice(ptr, m.length)
}

func (m *PinnedMemory[T]) Free() CudaError {
	m.valid = false
	err := C.free_host(&m.ptr)
	return CudaError(err)
}

func (m *PinnedMemory[T]) Valid() bool {
	return m.valid
}

func (m *PinnedMemory[T]) Size() uint {
	return m.size
}

func (m *PinnedMemory[T]) Length() uint {
	return m.length
}
