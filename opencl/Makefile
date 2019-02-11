CXX=g++ 
CFLAGS=-std=c++11 -O3 -l OpenCL -I${HOME}/GPU/include -I${HOME}/opt/include -I/apps/packages/cuda/10.0/include
LFLAGS=-l OpenCL -lboost_system -lboost_chrono -L${HOME}/opt/lib

all: test

test: main.o openclproduct.o fftopencl.o
	${CXX} main.o openclproduct.o fftopencl.o -o test ${LFLAGS}

main.o: main.cpp 
	${CXX} -c main.cpp ${CFLAGS}

fftopencl.o: fftopencl.hpp fftopencl.cpp
	${CXX} -c fftopencl.cpp ${CFLAGS}

openclproduct.o: openclproduct.hpp openclproduct.cpp
	${CXX} -c openclproduct.cpp ${CFLAGS}

clean:
	rm *.o test
