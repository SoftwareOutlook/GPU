#include "openaccproduct.hpp"


int openacc_product(const multiarray<complex>& z, const multiarray<double>& a, multiarray<complex>& az){

  unsigned long n=z.size();
  double* z_re=new double[n];
  double* z_im=new double[n];
  double* a_val=new double[n];
  double* az_re=new double[n];
  double* az_im=new double[n];

  unsigned long i;
  for(i=0; i<n; ++i){
    z_re[i]=z.get(i).real();
    z_im[i]=z.get(i).imag();
    a_val[i]=a.get(i);
  }

#pragma acc data copyin(n, z_re[0:n], z_im[0:n], a_val[0:n]) copyout(az_re[0:n], az_im[0:n]) 
#pragma acc kernels
  for(size_t i=0; i<n; ++i){
    az_re[i]=a_val[i]*z_re[i];
    az_im[i]=a_val[i]*z_im[i];
  }

  for(i=0; i<n; ++i){
    az.set(i, complex(az_re[i], az_im[i]));
  }

  delete[] z_re;
  delete[] z_im;
  delete[] a_val;
  delete[] az_re;
  delete[] az_im;  

  return 0;
}

int openacc_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax){

  unsigned long n=x.size();
  double* x_val=new double[n];
  double* a_val=new double[n];
  double* ax_val=new double[n];
  
  unsigned long i;
  for(i=0; i<n; ++i){
    x_val[i]=x.get(i);
    a_val[i]=a.get(i);
  }

#pragma acc data copyin(n, x_val[0:n], a_val[0:n]) copyout(ax_val[0:n]) 
#pragma acc kernels
  for(size_t i=0; i<n; ++i){
    ax_val[i]=a_val[i]*x_val[i];
  }

  for(i=0; i<n; ++i){
    ax.set(i, ax.get(i));
  }

  delete[] x_val;
  delete[] a_val;
  delete[] ax_val;  

  return 0;
}

