CXX=g++

CFLAGS=-std=c++11 -O3 -lboost_system -lboost_chrono 
LFLAGS=-lboost_system -lboost_chrono -fopenmp -L${HOME}/opt/lib


all: test

test: main.o 
	${CXX} main.o -o test ${LFLAGS} 

main.o: main.cpp 
	${CXX} -c main.cpp ${CFLAGS}

clean:
	rm *.o test
