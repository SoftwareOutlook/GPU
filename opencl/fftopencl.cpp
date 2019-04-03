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
  int i;
  for(i=0; i<n_dimensions(); ++i){
    n_x[i]=size(i);   
  }
  
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
  cl_context_properties context_properties[3];
  context_properties[0]=CL_CONTEXT_PLATFORM;
  context_properties[1]=(cl_context_properties)platform;
  context_properties[2]=0;
  int error;
  context=clCreateContext(context_properties, 1, &device, NULL, NULL, &error);
  queue=clCreateCommandQueue(context, device, 0, &error);
  
  clfftInitSetupData(&fft_setup);
  clfftSetup(&fft_setup);
  
}

fft_opencl::~fft_opencl(){   
  // clfftTeardown(); // Causes double free error, even if in main
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}



fft_opencl_c2c::fft_opencl_c2c(const std::vector<size_t>& i_dimensions, const bool i_inverse) : fft_opencl(i_dimensions, i_inverse){
    
  data=new double[2*size()+2];
  int error;
  d_data=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*size()*sizeof(double), NULL, &error);
  
  clfftDim dim;
  switch(n_dimensions()){
      case 1:
        dim=CLFFT_1D;
        break;
      case 2:
        dim=CLFFT_2D;
        break;
      case 3:
        dim=CLFFT_3D;
        break; 
      default:
        throw fft_opencl_exception("Works only in 1, 2 or 3 dimensions.");
  }
  clfftCreateDefaultPlan(&plan, context, dim, n_x);
  clfftSetPlanPrecision(plan, CLFFT_DOUBLE);
  clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
  clfftSetResultLocation(plan, CLFFT_INPLACE);
  clfftBakePlan(plan, 1, &queue, NULL, NULL);
 
}

fft_opencl_c2c::~fft_opencl_c2c(){
  delete[] data;
  clReleaseMemObject(d_data);
  clfftDestroyPlan(&plan);
}

int fft_opencl_c2c::compute(::complex* in, ::complex* out) const{

  size_t i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      data[2*i]=in[i].real();
      data[2*i+1]=in[i].imag();
    }
    clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0, 2*size()*sizeof(double),  data, 0, NULL, NULL);
    
    cl_command_queue queues[1];
    queues[0]=queue;
    clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, queues, 0, NULL, NULL, &d_data, NULL, NULL);
    clFinish(queue);
    
    clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, 2*size_complex()*sizeof(double),  data, 0, NULL, NULL);
    
    for(i=0; i<size_complex(); ++i){
      out[i].x=data[2*i];
      out[i].y=data[2*i+1];
    }
  }else{
    for(i=0; i<size_complex(); ++i){
      data[2*i]=out[i].real();
      data[2*i+1]=out[i].imag();
    }
    
    clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0,  2*size()*sizeof(double),  data, 0, NULL, NULL);
  
    cl_command_queue queues[1];
    queues[0]=queue;
    clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, queues, 0, NULL, NULL, &d_data, NULL, NULL);
    clFinish(queue);
    
    clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0,  2*size_complex()*sizeof(double),  data, 0, NULL, NULL);
    
    for(i=0; i<size(); ++i){
      in[i].x=data[2*i]/size();
      in[i].y=data[2*i+1]/size();
    }
  }
  return 0;
  
}


fft_opencl_r2c::fft_opencl_r2c(const std::vector<size_t>& i_dimensions, const bool i_inverse) : fft_opencl(i_dimensions, i_inverse){
  data=new double[2*size()+2];
  int error;
  d_data=clCreateBuffer(context, CL_MEM_READ_WRITE, (2*size()+2)*sizeof(double), NULL, &error);

  clfftDim dim;
  switch(n_dimensions()){
      case 1:
        dim=CLFFT_1D;
        break;
      case 2:
        dim=CLFFT_2D;
        break;
      case 3:
        dim=CLFFT_3D;
        break; 
      default:
        throw fft_opencl_exception("Works only in 1, 2 or 3 dimensions.");
  }
  clfftCreateDefaultPlan(&plan, context, dim, n_x);
  clfftSetPlanPrecision(plan, CLFFT_DOUBLE);
  clfftSetLayout(plan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  clfftSetResultLocation(plan, CLFFT_INPLACE);
  clfftBakePlan(plan, 1, &queue, NULL, NULL);
}

fft_opencl_r2c::~fft_opencl_r2c(){
  delete[] data;
  clReleaseMemObject(d_data);
  clfftDestroyPlan(&plan);
  
}

int fft_opencl_r2c::compute(double* in, ::complex* out) const{
    
  size_t i;
  if(!inverse){
    for(i=0; i<size(); ++i){
      data[i]=in[i];
    }
    clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0,  size()*sizeof(double),  data, 0, NULL, NULL);
    
    cl_command_queue queues[1];
    queues[0]=queue;
    clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, queues, 0, NULL, NULL, &d_data, NULL, NULL);
    clFinish(queue);
    
    
    clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, 2*size_complex()*sizeof(double),  data, 0, NULL, NULL);
    
    for(i=0; i<size_complex(); ++i){
      out[i].x=data[2*i];
      out[i].y=data[2*i+1];
}
  }else{
    for(i=0; i<size_complex(); ++i){
      data[2*i]=out[i].real();
      data[2*i+1]=out[i].imag();
    }
    
    clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0,  2*size_complex()*sizeof(double),  data, 0, NULL, NULL);
  
    cl_command_queue queues[1];
    queues[0]=queue;
    clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, queues, 0, NULL, NULL, &d_data, NULL, NULL);
    clFinish(queue);
    
    clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0,  size()*sizeof(double),  data, 0, NULL, NULL);
    
    for(i=0; i<size(); ++i){
      in[i]=data[i]/size();
    }
  }
  return 0;
    
}
