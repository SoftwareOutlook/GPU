#include "openclproduct.hpp"


class opencl_exception: public std::exception{
private:
  std::string message;
public:
  opencl_exception(const std::string& i_message){
    message=i_message;
  }   
  virtual const char* what() const throw(){     
    return message.c_str();   
  } 
}; 

opencl_exception opencl_init_exception("Error initialising OpenCL");


opencl_product::opencl_product(const int i_n_points, const int i_n_queues){

  n_points=i_n_points;
  
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if(platforms.empty()){
    throw opencl_init_exception;
  }
  platform=platforms.front();

  
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if(devices.size()<2){
    throw opencl_init_exception;
  }
  device=devices[1];

  context=cl::Context(std::vector<cl::Device>({device}));
 
  int i;
  for(i=0; i<i_n_queues; ++i){
    queues.push_back(cl::CommandQueue(context, device));
  }
}

opencl_product::~opencl_product(){
}



opencl_product_r::opencl_product_r(const int i_n_points, const int i_n_queues) : opencl_product(i_n_points, i_n_queues) {

  std::string kernel_source("__kernel void product(global double* x, global double* a, global double* ax){ unsigned int i = get_global_id(0); ax[i]=x[i]*a[i];}");

  cl::Program::Sources sources;

  sources.push_back({kernel_source.c_str(), kernel_source.length()});

  
  program=cl::Program(context, sources);
  program.build(std::vector<cl::Device>({device}));

  kernel=cl::Kernel(program, "product");

  x=new double[size()];
  a=new double[size()];
  ax=new double[size()];

  int i;
  
 
  d_x=cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size()*sizeof(double), x);
  d_a=cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size()*sizeof(double), a);
  d_ax=cl::Buffer(context, CL_MEM_WRITE_ONLY, size()*sizeof(double));
  
}


opencl_product_r::~opencl_product_r(){
  delete[] x;
  delete[] a;
  delete[] ax;
  for(int i=0; i<n_queues(); ++i){
    queues[i].finish();
  }
}

int opencl_product_r::compute(const multiarray<double>& i_x, const multiarray<double>& i_a, multiarray<double>& i_ax){

  int i;
  for(i=0; i<size(); ++i){
    x[i]=i_x.get(i);
    a[i]=i_a.get(i);
  }

  kernel.setArg(0, d_x);
  kernel.setArg(1, d_a);
  kernel.setArg(2, d_ax);

  for(i=0; i<n_queues(); ++i){
    queues[i].enqueueWriteBuffer(d_x, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), x+buffer_shift(i));
    queues[i].enqueueWriteBuffer(d_a, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), a+buffer_shift(i));
    queues[i].enqueueNDRangeKernel(kernel, cl::NDRange(buffer_shift(i)), cl::NDRange(buffer_size(i)));
    queues[i].enqueueReadBuffer(d_ax, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), ax+buffer_shift(i));
  }

  for(i=0; i<size(); ++i){
    i_ax.set(i, ax[i]);
  }
  return 0;

}


opencl_product_c::opencl_product_c(const int i_n_points, const int i_n_queues) : opencl_product(i_n_points, i_n_queues) {

  std::string kernel_source("__kernel void product(global double* z_re, global double* z_im, global double* a, global double* az_re, global double* az_im){ unsigned int i = get_global_id(0); az_re[i]=z_re[i]*a[i];  az_im[i]=z_im[i]*a[i];}");

  cl::Program::Sources sources;

  sources.push_back({kernel_source.c_str(), kernel_source.length()});

  
  program=cl::Program(context, sources);
  program.build(std::vector<cl::Device>({device}));

  kernel=cl::Kernel(program, "product");

  z_re=new double[size()];
  z_im=new double[size()];
  a=new double[size()];
  az_re=new double[size()];
  az_im=new double[size()];

  int i;
  
 
  d_z_re=cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size()*sizeof(double), z_re);
  d_z_im=cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size()*sizeof(double), z_im);
  d_a=cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size()*sizeof(double), a);
  d_az_re=cl::Buffer(context, CL_MEM_WRITE_ONLY, size()*sizeof(double));
  d_az_im=cl::Buffer(context, CL_MEM_WRITE_ONLY, size()*sizeof(double));
  
}


opencl_product_c::~opencl_product_c(){
  delete[] z_re;
  delete[] z_im;
  delete[] a;
  delete[] az_re;
  delete[] az_im;
  for(int i=0; i<n_queues(); ++i){
    queues[i].finish();
  }
}

int opencl_product_c::compute(const multiarray<complex>& i_z, const multiarray<double>& i_a, multiarray<complex>& i_az){

  int i;
  for(i=0; i<size(); ++i){
    z_re[i]=i_z.get(i).real();
    z_im[i]=i_z.get(i).imag();
    a[i]=i_a.get(i);
  }

  kernel.setArg(0, d_z_re);
  kernel.setArg(1, d_z_im);
  kernel.setArg(2, d_a);
  kernel.setArg(3, d_az_re);
  kernel.setArg(4, d_az_im);



  for(i=0; i<n_queues(); ++i){
    queues[i].enqueueWriteBuffer(d_z_re, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), z_re+buffer_shift(i));
    queues[i].enqueueWriteBuffer(d_z_im, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), z_im+buffer_shift(i));
    queues[i].enqueueWriteBuffer(d_a, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), a+buffer_shift(i));
    queues[i].enqueueNDRangeKernel(kernel, cl::NDRange(buffer_shift(i)), cl::NDRange(buffer_size(i)));
    queues[i].enqueueReadBuffer(d_az_re, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), az_re+buffer_shift(i));
    queues[i].enqueueReadBuffer(d_az_im, CL_TRUE, buffer_shift(i)*sizeof(double), sizeof(double)*buffer_size(i), az_im+buffer_shift(i));
  }

  for(i=0; i<size(); ++i){
    i_az.set(i, complex(az_re[i], az_im[i]));
  }
  return 0;

}
