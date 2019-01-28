#ifndef OPENCLPRODUCT_HPP
#define OPENCLPRODUCT_HPP

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <string>
#include <complex.hpp>
#include <multiarray.hpp>
#include <vector>

class opencl_product{
protected:
  
  int n_points;
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::Program program;
  cl::Kernel kernel;
  std::vector<cl::CommandQueue> queues;

public:
  opencl_product(const int i_n_points, const int i_n_queues=1);
  ~opencl_product();
  inline int size() const {
    return n_points;
  }
  inline int buffer_size(const int i) const {
    if(i<n_queues()-1){
      return size()/n_queues();
    }else{
      return size()-i*(size()/n_queues());
    }
  }
  inline int buffer_shift(const int i) const {
    return (size()/n_queues())*i;
  }
  inline int n_queues() const {
    return queues.size();
  }
};

class opencl_product_r : public opencl_product {
private:
  double *x, *a, *ax;
  cl::Buffer d_x, d_a, d_ax;
public:
  opencl_product_r(const int i_n_points, const int i_n_queues=1);
  ~opencl_product_r();
  int compute(const multiarray<double>& i_x, const multiarray<double>& i_a, multiarray<double>& i_ax); 
};

class opencl_product_c : public opencl_product {
private:
  double *z_re, *z_im, *a, *az_re, *az_im;
  cl::Buffer d_z_re, d_z_im, d_a, d_az_re, d_az_im;
public:
  opencl_product_c(const int i_n_points, const int i_n_queues=1);
  ~opencl_product_c();
  int compute(const multiarray<complex>& i_z, const multiarray<double>& i_a, multiarray<complex>& i_az);
};


#endif
