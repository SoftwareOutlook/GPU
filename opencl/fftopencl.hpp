#ifndef FFTOPENCL_HPP
#define FFTOPENCL_HPP

#include <vector>
#include "complex.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <CL/cl.h>
#include <CL/clFFT.h>


class fft_opencl{
public:
  typedef unsigned long size_t;
protected:
  std::vector<size_t> dimensions;
  bool inverse;
  cl_context context;
  cl_command_queue queue;
  unsigned long n_x[3];
  clfftPlanHandle plan;
  cl_mem buffer;
  mutable cl_complex *transform;
  mutable cl_mem d_transform;
public:
  fft_opencl(const std::vector<size_t>& i_dimensions, const bool i_inverse);
  ~fft_opencl();
  inline size_t n_dimensions() const {
    return dimensions.size();
  }
  inline size_t size(const size_t i) const {
    return dimensions[i];
  }
  inline size_t size() const {
    size_t i, n=1;
    for(i=0; i<n_dimensions(); ++i){
      n=n*size(i);
    }
    return n;
  }
  virtual size_t size_complex() const = 0;
};


class fft_opencl_c2c : public fft_opencl {
private:
  mutable cl_complex *signal;
  mutable cl_mem d_signal;
public:
  fft_opencl_c2c(const std::vector<size_t>& i_dimensions, const bool i_inverse=false);
  ~fft_opencl_c2c();
  inline size_t size_complex() const{
    return size();
  }
  int compute(::complex* in, ::complex* out) const;
};


class fft_opencl_r2c : public fft_opencl {
private:
  mutable cl_double *signal;
  mutable cl_mem d_signal;
public:
  fft_opencl_r2c(const std::vector<size_t>& i_dimensions, const bool i_inverse=false);
  ~fft_opencl_r2c();
  inline size_t size_complex() const{
    size_t i, n=1;
    for(i=0; i<n_dimensions()-1; ++i){
      n=n*size(i);
    }
    n=n*(size(n_dimensions()-1)/2+1);
    return n;
  }
  int compute(double* in, ::complex* out) const;
};

#endif
