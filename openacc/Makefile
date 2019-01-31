CXX=mpic++

FLAGS=-std=c++11 -O3 -fopenacc -lboost_system -lboost_chrono 


all: test

test: openaccproduct.o main.o 
	${CXX} openaccproduct.o main.o -o test ${FLAGS} 

main.o:  main.cpp 
	${CXX} -c main.cpp ${FLAGS}

openaccproduct.o:  openaccproduct.hpp openaccproduct.cpp 
	${CXX} -c openaccproduct.cpp ${FLAGS}
clean:
	rm *.o test
