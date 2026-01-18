.PHONY: clean

bin:
	mkdir bin

vectors-sum-lib: bin
	nvcc --shared --cudart=static -DCUDADLL_EXPORTS -o bin/vectors_sum.dll cu/vectors_sum.cu

build: vectors-sum-lib bin
	go build -o bin/main main.go

run:
	cd bin && ./main

clean:
	rm -rf bin



# nvcc -rdc=true --cudart=static --shared -DCUDADLL_EXPORTS -o bin/vectors_sum.dll cu/vectors_sum.cu

# 	nvcc -dc -c -o bin/temp.o cu/vectors_sum.cu
# 	nvcc -dlink -o bin/vectors_sum.dll bin/temp.o
# 	rm bin/temp.o

# 	nvcc -dc -O3 --cudart=static -o bin/vectors_sum cu/vectors_sum.cu
# 	nvcc -O3 --shared -Xcompiler -fPIC cu/vectors_sum.cu GlobalFunctions.so -o libCuFile.so
# 	nvcc -O3 --shared -o bin/vectors_sum.so cu/vectors_sum.cu
# $ nvcc   -rdc=true -c -o temp.o GPUFloydWarshall.cu
# $ nvcc -dlink -o GPUFloydWarshall.o temp.o -lcudart