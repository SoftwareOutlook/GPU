#include "fftcuda.hpp"


fft_cuda::fft_cuda(const std::vector<size_t>& i_dimensions, const bool i_inverse, cudaStream_t* i_stream) {
  dimensions=i_dimensions;
  inverse=i_inverse;
  stream=i_stream;
}

fft_cuda::~fft_cuda(){
  delete[] transform;
  cudaFree(d_transform);
  cufftDestroy(plan);
}



fft_cuda_c2c::fft_cuda_c2c(const std::vector<size_t>& i_dimensions, const bool i_inverse, cudaStream_t* i_stream) : fft_cuda(i_dimensions, i_inverse, i_stream){
  signal=new double2[size()];
  transform=new double2[size_complex()];
  cudaMalloc((void **) &d_signal, size()*sizeof(double2));
  cudaMalloc((void **) &d_transform, size_complex()*sizeof(double2));
  switch(n_dimensions()){
    case 1:
      cufftPlan1d(&plan, size(0), CUFFT_Z2Z, 1);
      break;      
    case 2:
      cufftPlan2d(&plan, size(0), size(1), CUFFT_Z2Z);
      break;
    case 3:
      cufftPlan3d(&plan, size(0), size(1), size(2), CUFFT_Z2Z);
      break;
    default:
      break;
  }
  if(stream!=nullptr){
    cufftSetStream(plan, *stream);
  }
}

fft_cuda_c2c::~fft_cuda_c2c(){
   delete[] signal;
   cudaFree(d_signal);
}

int fft_cuda_c2c::compute(::complex* in, ::complex* out) const{
  size_t i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      signal[i].x=in[i].real();
      signal[i].y=in[i].imag();
    }
    if(stream==nullptr){
      cudaMemcpy(d_signal, signal, size()*sizeof(double2), cudaMemcpyHostToDevice);
    }else{
      cudaMemcpyAsync(d_signal, signal, size()*sizeof(double2), cudaMemcpyHostToDevice, *stream);
    }
    cufftExecZ2Z(plan, d_signal, d_transform, CUFFT_FORWARD);  
    if(stream==nullptr){
      cudaMemcpy(transform, d_transform, size_complex()*sizeof(double2), cudaMemcpyDeviceToHost);
    }else{
      cudaMemcpyAsync(transform, d_transform, size_complex()*sizeof(double2), cudaMemcpyDeviceToHost, *stream);
    }
    for(i=0; i<size_complex(); ++i){
      out[i].x=transform[i].x;
      out[i].y=transform[i].y;
    }
  }else{
    for(i=0; i<size_complex(); ++i){
      transform[i].x=out[i].real();
      transform[i].y=out[i].imag();
    }
    if(stream==nullptr){
      cudaMemcpy(d_transform, transform, size_complex()*sizeof(double2), cudaMemcpyHostToDevice);
    }else{
      cudaMemcpyAsync(d_transform, transform, size_complex()*sizeof(double2), cudaMemcpyHostToDevice, *stream);
    }
    cufftExecZ2Z(plan, d_transform, d_signal, CUFFT_INVERSE);
    if(stream==nullptr){
      cudaMemcpy(signal, d_signal, size()*sizeof(double2), cudaMemcpyDeviceToHost);
    }else{
      cudaMemcpyAsync(signal, d_signal, size()*sizeof(double2), cudaMemcpyDeviceToHost, *stream);
    }
    for(i=0; i<size(); ++i){
      in[i].x=signal[i].x/size();
      in[i].y=signal[i].y/size();
    }
  }
  return 0;
}


fft_cuda_r2c::fft_cuda_r2c(const std::vector<size_t>& i_dimensions, const bool i_inverse, cudaStream_t* i_stream) : fft_cuda(i_dimensions, i_inverse, i_stream){
  signal=new double[size()];
  transform=new double2[size_complex()];
  cudaMalloc((void **) &d_signal, size()*sizeof(double));
  cudaMalloc((void **) &d_transform, size_complex()*sizeof(double2));
  cufftType type;
  if(!inverse){
    type=CUFFT_D2Z;
  }else{
    type=CUFFT_Z2D;
  }
  switch(n_dimensions()){
    case 1:
      cufftPlan1d(&plan, size(0), type, 1);
      break;      
    case 2:
      cufftPlan2d(&plan, size(0), size(1), type);
      break;
    case 3:
      cufftPlan3d(&plan, size(0), size(1), size(2), type);
      break;
    default:
      break;
  }
  if(stream!=nullptr){
    cufftSetStream(plan, *stream);
  }
}

fft_cuda_r2c::~fft_cuda_r2c(){
  delete[] signal;
  cudaFree(d_signal);
}

int fft_cuda_r2c::compute(double* in, ::complex* out) const{
  size_t i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      signal[i]=in[i];
    }
    if(stream==nullptr){
      cudaMemcpy(d_signal, signal, size()*sizeof(double), cudaMemcpyHostToDevice);
    }else{
      cudaMemcpyAsync(d_signal, signal, size()*sizeof(double), cudaMemcpyHostToDevice, *stream);
      }
    cufftExecD2Z(plan, d_signal, d_transform);
    if(stream==nullptr){
      cudaMemcpy(transform, d_transform, size_complex()*sizeof(double2), cudaMemcpyDeviceToHost);
    }else{
      cudaMemcpyAsync(transform, d_transform, size_complex()*sizeof(double2), cudaMemcpyDeviceToHost, *stream);
    }
    for(i=0; i<size_complex(); ++i){
      out[i].x=transform[i].x;
      out[i].y=transform[i].y;
    }
  }else{
    for(i=0; i<size_complex(); ++i){
      transform[i].x=out[i].real();
      transform[i].y=out[i].imag();
    }
    if(stream==nullptr){
      cudaMemcpy(d_transform, transform, size_complex()*sizeof(double2), cudaMemcpyHostToDevice);
    }else{
      cudaMemcpyAsync(d_transform, transform, size_complex()*sizeof(double2), cudaMemcpyHostToDevice, *stream);
    }
    cufftExecZ2D(plan, d_transform, d_signal);
    if(stream==nullptr){
      cudaMemcpy(signal, d_signal, size()*sizeof(double), cudaMemcpyDeviceToHost);
    }else{
      cudaMemcpyAsync(signal, d_signal, size()*sizeof(double), cudaMemcpyDeviceToHost, *stream);
    }
    for(i=0; i<size(); ++i){
      in[i]=signal[i]/size();
    }
  }
  return 0;
}
