#ifndef KOKKOS_HPP
#define KOKKOS_HPP

#include <multiarray.hpp>
#include <complex>
#include <Kokkos_Core.hpp>


class product_kokkos_complex{
private:
  int n;
  double *z_re, *z_im, *a_val, *az_re, *az_im;  
public:
  product_kokkos_complex(int i_n){
    n=i_n;
    z_re=new double[n];
    z_im=new double[n];
    a_val=new double[n];
    az_re=new double[n];
    az_im=new double[n];
  }
  ~product_kokkos_complex(){
    delete[] z_re;
    delete[] z_im;
    delete[] a_val;
    delete[] az_re;
    delete[] az_im; 
  }
  inline int size() const {
    return n;
  }
  int compute(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az){
    int i;
    for(i=0; i<n; ++i){
      z_re[i]=z.get(i).real();
      z_im[i]=z.get(i).imag();
      a_val[i]=a.get(i);
    }
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, n), KOKKOS_LAMBDA(const int& i){ 
      az_re[i]=a_val[i]*z_re[i];
      az_im[i]=a_val[i]*z_im[i];
    });
    for(i=0; i<n; ++i){
      az.set(i, ::complex(az_re[i], az_im[i]));
    }
    return 0;
  }
};


class product_kokkos_real{
private:
  int n;
  double *z_val, *a_val, *az_val;  
public:
  product_kokkos_real(int i_n){
    n=i_n;
    z_val=new double[n];
    a_val=new double[n];
    az_val=new double[n];
  }
  ~product_kokkos_real(){
    delete[] z_val;
    delete[] a_val;
    delete[] az_val; 
  }
  inline int size() const {
    return n;
  }
  int compute(const multiarray<double>& z, const multiarray<double>& a, multiarray<double>& az){
    int i;
    for(i=0; i<n; ++i){
      z_val[i]=z.get(i);
      a_val[i]=a.get(i);
    }
    // Kernel does not work with input classes to copies to an array are necessary
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, n), KOKKOS_LAMBDA(const int& i){ 
      az_val[i]=a_val[i]*z_val[i];
    });
    for(i=0; i<n; ++i){
      az.set(i, az_val[i]);
    }
    return 0;
  }
};


int kokkos_product_gpu(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az, Kokkos::View<double*, Kokkos::HostSpace>& h_z_re, Kokkos::View<double*, Kokkos::HostSpace>& h_z_im, Kokkos::View<double*, Kokkos::HostSpace>& h_a, Kokkos::View<double*, Kokkos::HostSpace>& h_az_re, Kokkos::View<double*, Kokkos::HostSpace>& h_az_im, Kokkos::View<double*, Kokkos::CudaSpace>& d_z_re, Kokkos::View<double*, Kokkos::CudaSpace>& d_z_im, Kokkos::View<double*, Kokkos::CudaSpace>& d_a, Kokkos::View<double*, Kokkos::CudaSpace>& d_az_re, Kokkos::View<double*, Kokkos::CudaSpace>& d_az_im){
  int n=z.size();
  int i;
  for(i=0; i<n; ++i){
    h_z_re(i)=z.get(i).real();
    h_z_im(i)=z.get(i).imag();
    h_a(i)=a.get(i);
  }
  Kokkos::deep_copy(d_z_re, h_z_re);
  Kokkos::deep_copy(d_z_im, h_z_im);
  Kokkos::deep_copy(d_a, h_a);
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, n) , KOKKOS_LAMBDA(const int& i){ 
    d_az_re(i)=d_a(i)*d_z_re(i);
    d_az_im(i)=d_a(i)*d_z_im(i);
  });
  Kokkos::deep_copy(h_az_re, d_az_re);
  Kokkos::deep_copy(h_az_im, d_az_im);
  for(i=0; i<n; ++i){
    az.set(i, ::complex(h_az_re(i), h_az_im(i)));
  }
  return 0;
 
}


int kokkos_product_gpu(const multiarray<double>& z, const multiarray<double>& a, multiarray<double>& az, Kokkos::View<double*, Kokkos::HostSpace>& h_z, Kokkos::View<double*, Kokkos::HostSpace>& h_a, Kokkos::View<double*, Kokkos::HostSpace>& h_az, Kokkos::View<double*, Kokkos::CudaSpace>& d_z, Kokkos::View<double*, Kokkos::CudaSpace>& d_a, Kokkos::View<double*, Kokkos::CudaSpace>& d_az){
  
  int n=z.size();
  int i;
  for(i=0; i<n; ++i){
    h_z(i)=z.get(i);
    h_a(i)=a.get(i);
  }
  Kokkos::deep_copy(d_z, h_z);
  Kokkos::deep_copy(d_a, h_a);
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, n) , KOKKOS_LAMBDA(const int& i){ 
    d_az(i)=d_a(i)*d_z(i);
  });
  Kokkos::deep_copy(h_az, d_az);
  for(i=0; i<n; ++i){
     az.set(i, h_az(i));
  }

  return 0;
  
}


#endif
