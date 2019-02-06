CXX=nvcc 

FLAGS=-I/home/pgambron/opt/include -I/home/pgambron/GPU/include -I/apps/packages/cuda/8.0/include -I/apps/packages/gcc/7.3.0/openmpi/3.1.2/include -L/apps/packages/gcc/7.3.0/openmpi/3.1.2/lib -L/home/pgambron/opt/lib -Xcompiler -fopenacc -laccfft -laccfft_utils -laccfft_gpu -laccfft_utils_gpu  -lfftw3 -lfftw3_omp -lfftw3_mpi -lfftw3_threads -lmpi -lcufft -Xcompiler -std=c++11 -lboost_system -lboost_chrono 

all: test

test: openaccproduct.o fftopenacc.o main.o 
	${CXX} openaccproduct.o fftopenacc.o main.o -o test ${FLAGS} 

main.o:  main.cpp 
	${CXX} -c main.cpp ${FLAGS}

fftopenacc.o: fftopenacc.hpp fftopenacc.cpp
	${CXX} -c fftopenacc.cpp ${FLAGS}

openaccproduct.o:  openaccproduct.hpp openaccproduct.cpp 
	${CXX} -c openaccproduct.cpp ${FLAGS}
clean:
	rm *.o test
