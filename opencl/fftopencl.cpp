#include "fftopencl.hpp"


class fft_opencl_exception: public std::exception{
private:
  std::string message;
public:
  fft_opencl_exception(const std::string& i_message){
    message=i_message;
  }   
  virtual const char* what() const throw(){     
    return message.c_str();   
  } 
};

fft_opencl::fft_opencl(const std::vector<size_t>& i_dimensions, const bool i_inverse) {
  dimensions=i_dimensions;
  inverse=i_inverse;
  for(i=0; i<n_dimensions(); ++i){
    n_x[i]=size(i);   
  }

  
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);
  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  
  
}

fft_opencl::~fft_opencl(){

}



fft_opencl_c2c::fft_opencl_c2c(const std::vector<size_t>& i_dimensions, const bool i_inverse) : fft_opencl(i_dimensions, i_inverse){
  int i;
  
  signal=(cl_complex*) malloc(size()*sizeof(cl_complex));
  transform=(cl_complex*) malloc(size_complex()*sizeof(cl_complex));
  
  context=clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  queue=clCreateCommandQueueWithProperties(context, device, 0, &err);

  clfftDim dim;
  switch(size()){
      case 1:
        dim=CLFFT_1D;
        break;
      case 2:
        dim=CLFFT_2D;
        break;
      case 1:
        dim=CLFFT_3D;
        break; 
      default:
        throw fft_opencl_exception("Works only in 1, 2 or 3 dimensions.");
  }
  
  clfftCreateDefaultPlan(&plan, context, dim, n_x);
  clfftSetPlanPrecision(plan, CLFFT_DOUBLE);
  clfftSetLayout(plan, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
  clfftSetResultLocation(plan, CLFFT_OUTOFPLACE);

  cl_int error;
  unsigned long buffer_size;
  clfftGetTmpBufSize(plan, &buffer_size);
  buffer=clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, 0, &error);
  
  d_signal=clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size()*sizeof(cl_complex), signal, &error);
  d_transform=clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_complex()*sizeof(cl_complex), signal, &error);
}

fft_opencl_c2c::~fft_opencl_c2c(){
  clfftDestroyPlan(&plan);
  clfftTeardown();
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseMemObject(buffer);
  clReleaseMemObject(d_signal);
  clReleaseMemObject(d_transform);
}

int fft_opencl_c2c::compute(::complex* in, ::complex* out) const{
    
  size_t i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      signal_re[i]=in[i].real();
      signal_im[i]=in[i].imag();
    }
    
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    
    clEnqueueWriteBuffer(queue, d_signal, CL_TRUE, 0,  size()*sizeof(cl_complex),  signal, 0, NULL, NULL);
        
    clfftEnqueueTransform(plan, CLFFT_FORWARD, n_dimensions(), &queue, 0, NULL, NULL, d_signal, d_transform, buffer);
  
    clEnqueueReadBuffer(queue, d_transform, CL_TRUE, 0,  size_complex()*sizeof(cl_complex),  transform, 0, NULL, NULL);
    
    for(i=0; i<size_complex(); ++i){
      out[i].x=transform[i][0];
      out[i].y=transform[i][1];
    }
  }else{
    for(i=0; i<size_complex(); ++i){
      transform[i][0]=out[i].real();
      transform[i][1]=out[i].imag();
    }
    
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    
    clEnqueueWriteBuffer(queue, d_transform, CL_TRUE, 0,  size_complex()*sizeof(cl_complex), transform, 0, NULL, NULL);
        
    clfftEnqueueTransform(plan, CLFFT_BACKWARD, n_dimensions(), &queue, 0, NULL, NULL, d_transform, d_signal, buffer);
  
    clEnqueueReadBuffer(queue, d_signal, CL_TRUE, 0,  size()*sizeof(cl_complex),  signal, 0, NULL, NULL);
    
    for(i=0; i<size(); ++i){
      in[i].x=signal[i][0]/size();
      in[i].y=signal[i][0]/size();
    }
  }
  return 0;
}


fft_opencl_r2c::fft_opencl_r2c(const std::vector<size_t>& i_dimensions, const bool i_inverse) : fft_opencl(i_dimensions, i_inverse){
  int i;
  
  signal=(cl_double*) malloc(size()*sizeof(cl_double));
  transform=(cl_complex*) malloc(size_complex()*sizeof(cl_complex));
  
  context=clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  queue=clCreateCommandQueueWithProperties(context, device, 0, &err);

  clfftDim dim;
  switch(size()){
      case 1:
        dim=CLFFT_1D;
        break;
      case 2:
        dim=CLFFT_2D;
        break;
      case 1:
        dim=CLFFT_3D;
        break; 
      default:
        throw fft_opencl_exception("Works only in 1, 2 or 3 dimensions.");
  }
  
  clfftCreateDefaultPlan(&plan, context, dim, n_x);
  clfftSetPlanPrecision(plan, CLFFT_DOUBLE);
  clfftSetLayout(plan, CLFFT_FFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  clfftSetResultLocation(plan, CLFFT_OUTOFPLACE);

  cl_int error;
  unsigned long buffer_size;
  clfftGetTmpBufSize(plan, &buffer_size);
  buffer=clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, 0, &error);
  
  d_signal=clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size()*sizeof(cl_double), signal, &error);
  d_transform=clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_complex()*sizeof(cl_complex), signal, &error);
  
}

fft_opencl_r2c::~fft_opencl_r2c(){
  clfftDestroyPlan(&plan);
  clfftTeardown();
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseMemObject(buffer);
  clReleaseMemObject(d_signal);
  clReleaseMemObject(d_transform);
}

int fft_opencl_r2c::compute(double* in, ::complex* out) const{
  size_t i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      signal[i]=in[i];
    }
    
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    
    clEnqueueWriteBuffer(queue, d_signal, CL_TRUE, 0,  size()*sizeof(cl_double),  signal, 0, NULL, NULL);
        
    clfftEnqueueTransform(plan, CLFFT_FORWARD, n_dimensions(), &queue, 0, NULL, NULL, d_signal, d_transform, buffer);
  
    clEnqueueReadBuffer(queue, d_transform, CL_TRUE, 0,  size_complex()*sizeof(cl_complex),  transform, 0, NULL, NULL);
    
    for(i=0; i<size_complex(); ++i){
      out[i].x=transform[i][0];
      out[i].y=transform[i][1];
    }
  }else{
    for(i=0; i<size_complex(); ++i){
      transform[i][0]=out[i].real();
      transform[i][1]=out[i].imag();
    }
    
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    
    clEnqueueWriteBuffer(queue, d_transform, CL_TRUE, 0,  size_complex()*sizeof(cl_complex), transform, 0, NULL, NULL);
        
    clfftEnqueueTransform(plan, CLFFT_BACKWARD, n_dimensions(), &queue, 0, NULL, NULL, d_transform, d_signal, buffer);
  
    clEnqueueReadBuffer(queue, d_signal, CL_TRUE, 0,  size()*sizeof(cl_double),  signal, 0, NULL, NULL);
    
    for(i=0; i<size(); ++i){
      in[i]=signal[i]/size();
    }
  }
  return 0;
}
