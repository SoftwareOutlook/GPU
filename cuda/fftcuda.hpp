#ifndef FFT_CUDA
#define FFT_CUDA

#include <vector>
#include "complex.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

class fft_cuda{
public:
  typedef unsigned long size_t;
protected:
  std::vector<size_t> dimensions;
  mutable double2 *transform, *d_transform;
  cufftHandle plan;
  bool inverse;
  cudaStream_t *stream;
public:
  fft_cuda(const std::vector<size_t>& i_dimensions, const bool i_inverse, cudaStream_t* i_stream);
  ~fft_cuda();
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


class fft_cuda_c2c : public fft_cuda {
private:
  mutable double2 *signal, *d_signal;
public:
  fft_cuda_c2c(const std::vector<size_t>& i_dimensions, const bool i_inverse=false, cudaStream_t* i_stream=nullptr);
  ~fft_cuda_c2c();
  inline size_t size_complex() const{
    return size();
  }
  int compute(::complex* in, ::complex* out) const;
};


class fft_cuda_r2c : public fft_cuda {
private:
  mutable double *signal, *d_signal;
public:
  fft_cuda_r2c(const std::vector<size_t>& i_dimensions, const bool i_inverse=false, cudaStream_t* i_stream=nullptr);
  ~fft_cuda_r2c();
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
