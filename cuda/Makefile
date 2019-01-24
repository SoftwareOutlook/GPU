CXX=nvcc

CFLAGS=-std=c++11 -O3 -lboost_system -lboost_chrono 
LFLAGS=-lboost_system -lboost_chrono -lcufft -Xcompiler -fopenmp


all: test

test: cudaproduct.o fftcuda.o main.o 
	${CXX} cudaproduct.o fftcuda.o main.o -o test ${LFLAGS} 

main.o: openmpproduct.hpp main.cpp 
	${CXX} -c main.cpp ${CFLAGS}

cudaproduct.o: cudaproduct.cuh cudaproduct.cu
	${CXX} -c cudaproduct.cu ${CFLAGS}

fftcuda.o: fftcuda.hpp fftcuda.cpp
	${CXX} -c fftcuda.cpp ${CFLAGS}


clean:
	rm *.o test
