CXX=g++

CFLAGS=-std=c++11 -O3 -lboost_system -lboost_chrono -I${HOME}/GPU/include -I${HOME}/opt/include
LFLAGS=-lboost_system -lboost_chrono -fopenmp -L${HOME}/opt/lib -lkokkos -ldl


all: test

test: kokkos.o main.o 
	${CXX} kokkos.o main.o -o test ${LFLAGS} 

main.o: main.cpp 
	${CXX} -c main.cpp ${CFLAGS}

kokkos.o: kokkos.hpp kokkos.cpp
	${CXX} -c kokkos.cpp ${CFLAGS}

clean:
	rm *.o test
