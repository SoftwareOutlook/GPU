#ifndef CUDA_CUH
#define CUDA_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <complex.hpp>
#include <multiarray.hpp>
#include <cufft.h>

__global__ void cuda_product_kernel(const int n, double* z_re, double* z_im, double* a, double* az_re, double* az_im);

__global__ void cuda_product_kernel(const int n, double* x, double* a, double* ax);

void cuda_product_part(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az, const unsigned long shift, cudaStream_t& stream);

void cuda_product_part(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax, const unsigned long shift, cudaStream_t& stream);

void cuda_product(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az, const unsigned long n_streams);

void cuda_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax, const unsigned long n_streams);

void cuda_product(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az);

void cuda_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax);

#endif
