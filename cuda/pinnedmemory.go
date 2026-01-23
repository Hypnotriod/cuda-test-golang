package cuda

/*
#include <stdlib.h>
#include <stdint.h>

int malloc_host(uint8_t **ptr, size_t size);
int free_host(uint8_t **ptr);
*/
import "C"
import (
	"reflect"
	"unsafe"
)

type PinnedMemory[T any] struct {
	p       *C.uint8_t
	length  int
	invalid bool
}

func PinnedMemoryNew[T any](length int) (*PinnedMemory[T], CudaError) {
	var v T
	t := reflect.TypeOf(v)
	m := &PinnedMemory[T]{
		length: length,
	}
	err := C.malloc_host(&m.p, C.size_t(length*int(t.Size())))
	return m, CudaError(err)
}

func (m *PinnedMemory[T]) Slice() []T {
	if m.invalid {
		var slice []T
		return slice
	}
	var array *T = (*T)(unsafe.Pointer(m.p))
	return unsafe.Slice(array, m.length)
}

func (m *PinnedMemory[T]) Free() CudaError {
	m.invalid = true
	err := C.free_host(&m.p)
	return CudaError(err)
}
