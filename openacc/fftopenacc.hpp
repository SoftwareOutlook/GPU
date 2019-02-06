#ifndef FFTOPENACC_HPP
#define FFTOPENACC_HPP

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <accfft.h>
#include <accfft_gpu.h>
#include <mpi.h>
#include <complex.hpp>
#include <math.h>

class fft_openacc{
protected:
  std::vector<int> dimensions;
  bool inverse;
  MPI_Comm accfft_communicator;
  accfft_plan_gpu* plan;
public:
  fft_openacc(const std::vector<int>& dimensions, const bool i_inverse);
  ~fft_openacc();
  inline int n_dimensions() const {
    return dimensions.size();
  }
  inline int size(const int i) const {
    return dimensions[i];
  }
  inline int size() const {
    int i, n=1;
    for(i=0; i<n_dimensions(); ++i){
      n=n*size(i);
    }
    return n;
  }
  virtual int size_complex() const = 0;
};

class fft_openacc_c2c : public fft_openacc {
private:
  mutable Complex *signal, *d_signal, *transform, *d_transform;
public:
  fft_openacc_c2c(const std::vector<int>& i_dimensions, const bool i_inverse=false);
  ~fft_openacc_c2c();
  int compute(complex* in, complex* out);
  inline int size_complex() const {
    return size();   
  }
};

class fft_openacc_r2c : public fft_openacc {
private:
  mutable double *signal, *d_signal;
  mutable Complex *transform, *d_transform;
public:
  fft_openacc_r2c(const std::vector<int>& i_dimensions, const bool i_inverse=false);
  ~fft_openacc_r2c();
  
  int compute(double* in, complex* out);
  inline int size_complex() const {
    return ceil((double)size()/2);   
  }
};

#endif
