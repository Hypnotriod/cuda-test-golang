package cuda

/*
#cgo LDFLAGS: -lvectors_sum -L${SRCDIR}/../bin

#include <stdlib.h>
#include <stdint.h>

extern int vectors_sum(uint32_t *a, uint32_t *b, uint32_t N);
*/
import "C"

func VectorsSum(a []uint32, b []uint32) CudaError {
	err := C.vectors_sum((*C.uint)(&a[0]), (*C.uint)(&b[0]), (C.uint)(min(len(a), len(b))))
	return CudaError(err)
}
