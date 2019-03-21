#ifndef FFTOPENCL_HPP
#define FFTOPENCL_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <vector>
#include "complex.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <CL/cl.h>
#include <CL/clFFT.h>
#include <CL/cl.hpp>


class fft_opencl{
public:
  typedef unsigned long size_t;
protected:
  std::vector<size_t> dimensions;
  bool inverse;
  unsigned long n_x[3];
  
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  
  clfftSetupData fft_setup;

  mutable double* data;
  mutable cl_mem d_data;
    
  clfftPlanHandle plan;
  
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
  virtual size_t size_complex(const size_t i) const = 0;
  virtual size_t size_complex() const = 0;
};


class fft_opencl_c2c : public fft_opencl {
public:
  fft_opencl_c2c(const std::vector<size_t>& i_dimensions, const bool i_inverse=false);
  ~fft_opencl_c2c();
  inline size_t size_complex(const size_t i) const override {
    return size(i);   
  }
  inline size_t size_complex() const{
    size_t i, n=1;
    for(i=0; i<n_dimensions(); ++i){
      n=n*size_complex(i);
    }
    return n;   
  }
  int compute(::complex* in, ::complex* out) const;
};


class fft_opencl_r2c : public fft_opencl {
public:
  fft_opencl_r2c(const std::vector<size_t>& i_dimensions, const bool i_inverse=false);
  ~fft_opencl_r2c();  
  inline size_t size_complex(const size_t i) const override {
    if(i<(n_dimensions()-1)){
      return size(i);
    }else{
      return ceil(((double)size(i))/2);  
    }
  }
  inline size_t size_complex() const{
    size_t i, n=1;
    for(i=0; i<n_dimensions(); ++i){
      n=n*size_complex(i);
    }
    return n;   
  }
  int compute(double* in, ::complex* out) const;
};

#endif
