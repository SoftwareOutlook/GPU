#include "cudaproduct.cuh"
#include <cufft.h>
#include <omp.h>

#define N_THREADS_PER_KERNEL 32


__global__ void cuda_product_kernel(const int n, double* z_re, double* z_im, double* a, double* az_re, double* az_im){
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  int stride=blockDim.x*gridDim.x;
  for (int i=index; i<n; i+=stride){
    az_re[i]=a[i]*z_re[i];
    az_im[i]=a[i]*z_im[i];
  }
}

__global__ void cuda_product_kernel(const int n, double* x, double* a, double* ax){
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  int stride=blockDim.x*gridDim.x;
  for (int i=index; i<n; i+=stride){
    ax[i]=a[i]*x[i];
  }
}

void cuda_product_part(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az, const unsigned long n, const unsigned long shift, cudaStream_t& stream){
  double* z_re=new double[n];
  double* z_im=new double[n];
  double* a_val=new double[n];
  double* az_re=new double[n];
  double* az_im=new double[n];

  unsigned long i;
  for(i=0; i<n; ++i){
    z_re[i]=z.get(shift+i).real();
    z_im[i]=z.get(shift+i).imag();
    a_val[i]=a.get(shift+i);
  }

  double *d_z_re, *d_z_im, *d_a_val, *d_az_re, *d_az_im;
  cudaMalloc((void**)&d_z_re, n*sizeof(double));
  cudaMalloc((void**)&d_z_im, n*sizeof(double));
  cudaMalloc((void**)&d_a_val, n*sizeof(double));
  cudaMalloc((void**)&d_az_re, n*sizeof(double));
  cudaMalloc((void**)&d_az_im, n*sizeof(double));

  cudaMemcpyAsync(d_z_re, z_re, n*sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_z_im, z_im, n*sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_a_val, a_val, n*sizeof(double), cudaMemcpyHostToDevice, stream);

  cuda_product_kernel<<<(n+N_THREADS_PER_KERNEL-1)/N_THREADS_PER_KERNEL, N_THREADS_PER_KERNEL, 0, stream>>>(n, d_z_re, d_z_im, d_a_val, d_az_re, d_az_im);
  
  cudaMemcpyAsync(az_re, d_az_re, n*sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(az_im, d_az_im, n*sizeof(double), cudaMemcpyDeviceToHost, stream);

  cudaFree(d_z_re);
  cudaFree(d_z_im);
  cudaFree(d_a_val);
  cudaFree(d_az_re);
  cudaFree(d_az_im);

  for(i=0; i<n; ++i){
    az.set(shift+i, complex(az_re[i], az_im[i]));
  }

  delete[] z_re;
  delete[] z_im;
  delete[] a_val;
  delete[] az_re;
  delete[] az_im;
}

void cuda_product_part(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax, const unsigned long n, const unsigned long shift, cudaStream_t& stream){
  double* x_val=new double[n];
  double* a_val=new double[n];
  double* ax_val=new double[n];

  unsigned long i;
  for(i=0; i<n; ++i){
    x_val[i]=x.get(shift+i);
    a_val[i]=a.get(shift+i);
  }

  double *d_x_val, *d_a_val, *d_ax_val;
  cudaMalloc((void**)&d_x_val, n*sizeof(double));
  cudaMalloc((void**)&d_a_val, n*sizeof(double));
  cudaMalloc((void**)&d_ax_val, n*sizeof(double));

  cudaMemcpyAsync(d_x_val, x_val, n*sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_a_val, a_val, n*sizeof(double), cudaMemcpyHostToDevice, stream);

  cuda_product_kernel<<<(n+N_THREADS_PER_KERNEL-1)/N_THREADS_PER_KERNEL, N_THREADS_PER_KERNEL, 0, stream>>>(n, d_x_val, d_a_val, d_ax_val);
  
  cudaMemcpyAsync(ax_val, d_ax_val, n*sizeof(double), cudaMemcpyDeviceToHost, stream);

  cudaFree(d_x_val);
  cudaFree(d_a_val);
  cudaFree(d_ax_val);

  for(i=0; i<n; ++i){
    ax.set(shift+i, ax_val[i]);
  }

  delete[] x_val;
  delete[] a_val;
  delete[] ax_val;
}

void cuda_product(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az, const unsigned long n_streams){
  cudaStream_t streams[n_streams];
  unsigned int i, n=z.size()/n_streams;
#pragma omp parallel for
  for(i=0; i<n_streams; ++i){
    cudaStreamCreate(streams+i);
    if(i<n_streams-1){
      cuda_product_part(z, a, az, n, i*n, streams[i]);
    }else{
      cuda_product_part(z, a, az, n+z.size()%n_streams, i*n, streams[i]);
    }
  }
  cudaDeviceSynchronize();
}

void cuda_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax, const unsigned long n_streams){
  cudaStream_t streams[n_streams];
  unsigned int i, n=x.size()/n_streams;
#pragma omp parallel for
  for(i=0; i<n_streams; ++i){
    cudaStreamCreate(streams+i);
    if(i<n_streams-1){
      cuda_product_part(x, a, ax, n, i*n, streams[i]);
    }else{
      cuda_product_part(x, a, ax, n+x.size()%n_streams, i*n, streams[i]);
    }
  }
  cudaDeviceSynchronize();
}















void cuda_product(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az){
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

  double *d_z_re, *d_z_im, *d_a_val, *d_az_re, *d_az_im;
  cudaMalloc((void**)&d_z_re, n*sizeof(double));
  cudaMalloc((void**)&d_z_im, n*sizeof(double));
  cudaMalloc((void**)&d_a_val, n*sizeof(double));
  cudaMalloc((void**)&d_az_re, n*sizeof(double));
  cudaMalloc((void**)&d_az_im, n*sizeof(double));

  cudaMemcpy(d_z_re, z_re, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z_im, z_im, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_val, a_val, n*sizeof(double), cudaMemcpyHostToDevice);

  cuda_product_kernel<<<(n+N_THREADS_PER_KERNEL-1)/N_THREADS_PER_KERNEL, N_THREADS_PER_KERNEL>>>(n, d_z_re, d_z_im, d_a_val, d_az_re, d_az_im);
  
  cudaMemcpyAsync(az_re, d_az_re, n*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(az_im, d_az_im, n*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_z_re);
  cudaFree(d_z_im);
  cudaFree(d_a_val);
  cudaFree(d_az_re);
  cudaFree(d_az_im);

  for(i=0; i<n; ++i){
    az.set(i, complex(az_re[i], az_im[i]));
  }

  delete[] z_re;
  delete[] z_im;
  delete[] a_val;
  delete[] az_re;
  delete[] az_im;
}

void cuda_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax){
  unsigned long n=x.size();
  double* x_val=new double[n];
  double* a_val=new double[n];
  double* ax_val=new double[n];

  unsigned long i;
  for(i=0; i<n; ++i){
    x_val[i]=x.get(i);
    a_val[i]=a.get(i); 
  }

  double *d_x_val, *d_a_val, *d_ax_val;
  cudaMalloc((void**)&d_x_val, n*sizeof(double));
  cudaMalloc((void**)&d_a_val, n*sizeof(double));
  cudaMalloc((void**)&d_ax_val, n*sizeof(double));

  cudaMemcpy(d_x_val, x_val, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_val, a_val, n*sizeof(double), cudaMemcpyHostToDevice);

  cuda_product_kernel<<<(n+N_THREADS_PER_KERNEL-1)/N_THREADS_PER_KERNEL, N_THREADS_PER_KERNEL>>>(n, d_x_val, d_a_val, d_ax_val);
  
  cudaMemcpy(ax_val, d_ax_val, n*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x_val);
  cudaFree(d_a_val);
  cudaFree(d_ax_val);

  for(i=0; i<n; ++i){
    ax.set(i, ax_val[i]);
  }

  delete[] x_val;
  delete[] a_val;
  delete[] ax_val;
}