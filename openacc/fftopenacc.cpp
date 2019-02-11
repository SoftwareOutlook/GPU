#include "fftopenacc.hpp"

fft_openacc::fft_openacc(const std::vector<int>& i_dimensions, const bool i_inverse){
  dimensions=i_dimensions;
  inverse=i_inverse;
  
  int accfft_dimensions[2]={1, 1};
  accfft_create_comm(MPI_COMM_WORLD, accfft_dimensions, &accfft_communicator);
}

fft_openacc::~fft_openacc(){

}

fft_openacc_c2c::fft_openacc_c2c(const std::vector<int>& i_dimensions, const bool i_inverse) : fft_openacc(i_dimensions, i_inverse){
  int i, n_x[3], input_size[3], input_start[3], output_size[3], output_start[3];
  for(i=0; i<n_dimensions(); ++i){
    n_x[i]=dimensions[i];   
  }
  for( ; i<3; ++i){
    n_x[i]=1;   
  }
  
  int size_max=accfft_local_size_dft_c2c_gpu(n_x, input_size, input_start, output_size, output_start, accfft_communicator);
  
  signal=(Complex*) accfft_alloc(size_max);
  transform=(Complex*) accfft_alloc(size_max);
  accfft_init(1);
  cudaMalloc((void**) &d_signal, size()*sizeof(Complex));
  cudaMalloc((void**) &d_transform, size()*sizeof(Complex));
  plan=accfft_plan_dft_3d_c2c_gpu(n_x, d_signal, d_transform, accfft_communicator, ACCFFT_ESTIMATE);
}

fft_openacc_c2c::~fft_openacc_c2c(){
  accfft_destroy_plan(plan);
  accfft_free(signal);
  accfft_free(transform);
  cudaFree(d_signal);
  cudaFree(d_transform);
}

int fft_openacc_c2c::compute(complex* in, complex* out){
  int i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      signal[i][0]=in[i].real();
      signal[i][1]=in[i].imag();
    }
  }else{
    for(i=0; i<size(); ++i){
      signal[i][0]=out[i].real();
      signal[i][1]=out[i].imag();
    }      
  }
  cudaMemcpy(d_signal, signal, size()*sizeof(Complex), cudaMemcpyHostToDevice);
  if(!inverse){
    accfft_execute_c2c_gpu(plan, ACCFFT_FORWARD, d_signal, d_transform);
  }else{
    accfft_execute_c2c_gpu(plan, ACCFFT_BACKWARD, d_signal, d_transform);
  }
  cudaMemcpy(transform, d_transform, size()*sizeof(Complex), cudaMemcpyDeviceToHost);
  if(!inverse){
    for(i=0; i<size(); ++i){
      out[i]=complex(transform[i][0], transform[i][1]);
    }
  }else{
    for(i=0; i<size(); ++i){
      in[i]=complex(transform[i][0], transform[i][1])/size();
    }
  }
  return 0;
}

fft_openacc_r2c::fft_openacc_r2c(const std::vector<int>& i_dimensions, const bool i_inverse) : fft_openacc(i_dimensions, i_inverse){
  int i, n_x[3], input_size[3], input_start[3], output_size[3], output_start[3];
  for(i=0; i<n_dimensions(); ++i){
    n_x[i]=dimensions[i];   
  }
  for( ; i<3; ++i){
    n_x[i]=1;   
  }
  int size_max;
  size_max=accfft_local_size_dft_r2c_gpu(n_x, input_size, input_start, output_size, output_start, accfft_communicator);
  
  signal=(double*) accfft_alloc(size_max);
  transform=(Complex*) accfft_alloc(size_max);
  accfft_init(1);
  cudaMalloc((void**) &d_signal, size()*sizeof(double));
  cudaMalloc((void**) &d_transform, size_max*sizeof(double));
  plan=accfft_plan_dft_3d_r2c_gpu(n_x, d_signal, (double*) d_transform, accfft_communicator, ACCFFT_ESTIMATE);
}

fft_openacc_r2c::~fft_openacc_r2c(){
  accfft_destroy_plan(plan);
  accfft_free(signal);
  accfft_free(transform);
  cudaFree(d_signal);
  cudaFree(d_transform);
}

int fft_openacc_r2c::compute(double* in, complex* out){
 
  int i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      signal[i]=in[i];
    }
    cudaMemcpy(d_signal, signal, size()*sizeof(double), cudaMemcpyHostToDevice);    
    accfft_execute_r2c_gpu(plan, d_signal, d_transform);
    cudaMemcpy(transform, d_transform, size_complex()*sizeof(Complex), cudaMemcpyDeviceToHost);
    for(i=0; i<size_complex(); ++i){
      out[i]=complex(transform[i][0], transform[i][1]);
    }
  }else{
    for(i=0; i<size_complex(); ++i){
      transform[i][0]=out[i].real();
      transform[i][1]=out[i].imag();
    }
	cudaMemcpy(d_transform, transform, size_complex() * sizeof(Complex), cudaMemcpyHostToDevice);
    accfft_execute_c2r_gpu(plan, d_transform, d_signal);
    cudaMemcpy(signal, d_signal, size() * sizeof(double), cudaMemcpyDeviceToHost);
    for(i=0; i<size(); ++i){
      in[i]=signal[i]/size();
    }
  }
  return 0;
}
