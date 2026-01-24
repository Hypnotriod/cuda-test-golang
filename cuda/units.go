package cuda

/*
#cgo LDFLAGS: -llibrary -L${SRCDIR}/../bin

#include <stdlib.h>
#include <stdint.h>

int vector_add_uint32(uint32_t *a, uint32_t *b, uint32_t N);
*/
import "C"

func VectorAddUint32(a []uint32, b []uint32) CudaError {
	err := C.vector_add_uint32((*C.uint32_t)(&a[0]), (*C.uint32_t)(&b[0]), (C.uint32_t)(min(len(a), len(b))))
	return CudaError(err)
}
