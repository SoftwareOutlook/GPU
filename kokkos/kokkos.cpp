#include "kokkos.hpp"
#include <Kokkos_Core.hpp>
using namespace Kokkos;

void kokkos_loop(const int n, double* z_re, double* z_im, double* a_val, double* az_re, double* az_im){

  int argc=0;
  Kokkos::initialize(argc, nullptr);
  
  parallel_for("product", n, KOKKOS_LAMBDA(const int& i){ 
  az_re[i]=a_val[i]*z_re[i];
  az_im[i]=a_val[i]*z_im[i];
  });

  Kokkos::finalize();
}


void kokkos_loop(const int n, double* x, double* a, double* ax){

  int argc=0;
  Kokkos::initialize(argc, nullptr);
  
  parallel_for("product", n, KOKKOS_LAMBDA(const int& i){ 
  ax[i]=a[i]*x[i];
  });

  Kokkos::finalize();
}

int kokkos_product(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az){
  
  unsigned long n=z.size();
  double* z_re=new double[n];
  double* z_im=new double[n];
  double* a_val=new double[n];
  double* az_re=new double[n];
  double* az_im=new double[n];
  
  kokkos_loop(n, z_re, z_im, a_val, az_re, az_im);
  
  delete[] z_re;
  delete[] z_im;
  delete[] a_val;
  delete[] az_re;
  delete[] az_im; 
  
  return 0;
}

int kokkos_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax){
    
  unsigned long n=x.size();
  double* x_val=new double[n];
  double* a_val=new double[n];
  double* ax_val=new double[n];
  
  kokkos_loop(n, x_val, a_val, ax_val);
  
  delete[] x_val;
  delete[] a_val;
  delete[] ax_val;

  return 0;
}
