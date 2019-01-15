CXX=g++
CC=gcc
FLAGS=-std=c++11 -lboost_system -lboost_chrono

all: test

test: main.cpp 
	${CXX} main.cpp -o test ${FLAGS}  

clean:
	rm *.o test
