.PHONY: build run clean cuda-library 

bin:
	mkdir bin

cuda-library: bin
	nvcc -O3 --shared --cudart=static -DCUDADLL_EXPORTS -o bin/library.dll cu/library.cu

build: cuda-library bin
	go build -o bin/main.exe main.go

run:
	cd bin && ./main.exe

clean:
	rm -rf bin
