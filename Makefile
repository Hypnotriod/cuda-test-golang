.PHONY: build run clean cuda-library
default: build

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	LIBFILE := liblibrary.so
endif
ifeq ($(UNAME_S),Darwin)
	LIBFILE := liblibrary.dylib
endif
ifeq ($(OS),Windows_NT)
	LIBFILE := library.dll
endif

bin:
	mkdir -p bin

cuda-library: bin
	nvcc -O3 --shared --cudart=static -DCUDADLL_EXPORTS -o bin/$(LIBFILE) cu/library.cu

build: cuda-library bin
	go build -o bin/main.exe main.go

run:
	cd bin && ./main.exe

clean:
	rm -rf bin
