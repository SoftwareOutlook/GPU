CXX=g++ 
CFLAGS=-std=c++11 -O3 -l OpenCL 
LFLAGS=-l OpenCL -lboost_system -lboost_chrono

all: test

test: main.o openclproduct.o
	${CXX} main.o openclproduct.o -o test ${LFLAGS}

main.o: main.cpp 
	${CXX} -c main.cpp ${CFLAGS}

openclproduct.o: openclproduct.hpp openclproduct.cpp
	${CXX} -c openclproduct.cpp ${CFLAGS}

clean:
	rm *.o test
