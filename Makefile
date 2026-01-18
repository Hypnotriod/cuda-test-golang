.PHONY: clean

bin:
	mkdir bin

build-cuda-libs: bin
	nvcc -O3 --shared --cudart=static -DCUDADLL_EXPORTS -o bin/vectors_sum.dll cu/vectors_sum.cu

build: build-cuda-libs bin
	go build -o bin/main main.go

run:
	cd bin && ./main

clean:
	rm -rf bin
