#include <complex.hpp>
#include <signal.hpp>
#include <iostream>
#include <stopwatch.hpp>
#include <multiarray.hpp>
#include <vector>
#include "openclproduct.hpp"
#include <omp.h>
#include "fftopencl.hpp"

using namespace std;


int main(int argc, char** argv){

    
  if(argc<5){
    std::cout << "Please supply 3 dimensions and a number of coils\n";
    return -1;
  }

  
  int i, j, k[3], n_x[3], dim[3], n_dimensions, n_coils;
  double a=0.25;
  double b=a/2;
  double s, x[3], l_x[3]={1, 1, 1}, error;
  for(i=0; i<3; ++i){
    dim[i]=atoi(*(argv+i+1));
  }
  n_coils=atoi(*(argv+4));
  
  
  stopwatch sw;


  // 1D
  
  n_dimensions=1;
  n_x[0]=dim[0];
  
  // R <-> HC
  
  {
    // Dimensions
    
    std::cout << "Dimensions: ";
    for(i=0; i<n_dimensions; ++i){
      std::cout << n_x[i] << " ";
    }
    std::cout << "\n";
    std::cout << "N coils: " << n_coils << "\n";
    std::cout << "\n";
    std::cout << "  R <-> HC\n";
    std::cout << "\n";
    
    // Signal
    multiarray<double> sig({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      s=signal(n_dimensions, l_x, a, b, x);
      sig(k[0])=s;
    }
    
    // Coils
    multiarray<double> coil({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      coil(k[0])=1./n_coils;
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
   
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      // multiplied_signals.push_back(sig*coils[i]);
      multiplied_signals.push_back(multiarray<double>({n_x[0]}));
      transforms.push_back(multiarray<::complex>({ceil(((double)n_x[0])/2)}));
      inverse_transforms.push_back(multiarray<double>({n_x[0]}));
    }

    std::cout << "    Multiplication\n";

    // OpenCL

    int n_queues=1;
    std::cout << "      OpenCL (" << n_queues << " queues)\n";
    opencl_product_r opr1(sig.size(), n_queues);
    sw.start();
    for(i=0; i<n_coils; ++i){
      opr1.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";  

    
    n_queues=4;
    std::cout << "      OpenCL (" << n_queues << " queues)\n";
    opencl_product_r opr(sig.size(), n_queues);
    sw.start();
    for(i=0; i<n_coils; ++i){
      opr.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";  

//     
    // FFT
    std::cout << "    FFT\n";
    
    // clFFT
    std::cout << "      clFFT\n";
    std::cout << "        Direct\n";

    fft_opencl_r2c fo({n_x[0]});
    
     
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      fo.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();

    std::cout << "          Time:  " << sw.get() << " s\n";    

    std::cout << "        Inverse\n";

    fft_opencl_r2c foi({n_x[0]}, true);
 
    sw.start();
    for(i=0; i<n_coils; ++i){
      foi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "        Error: " << error << "\n";
    std::cout << "\n";
    
  }

  
  // C <-> C
  
  {
    // Dimensions
    
    std::cout << "  C <-> C\n";
    std::cout << "\n";
    
    // Signal
    multiarray<::complex> sig({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      s=signal(n_dimensions, l_x, a, b, x);
      sig(k[0])=::complex(s, s);
    }
    
    // Coils
    multiarray<double> coil({n_x[0]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      coil(k[0])=1./n_coils;
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }

   

    
    // Multiplied signals
    std::vector<multiarray<::complex>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<::complex>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]);
      transforms.push_back(multiarray<::complex>({n_x[0]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0]}));
    }


    int n_queues=1;
    std::cout << "      OpenCL (" << n_queues << " queues)\n";
    opencl_product_c opc1(sig.size(), n_queues);
    sw.start();
    for(i=0; i<n_coils; ++i){
      opc1.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    n_queues=4;
    std::cout << "      OpenCL (" << n_queues << " queues)\n";
    opencl_product_c opc(sig.size(), n_queues);
    sw.start();
    for(i=0; i<n_coils; ++i){
      opc.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";


    
    
    // FFT
    std::cout << "    FFT\n";
    // OpenCL
    std::cout << "      OpenCL\n";
    std::cout << "        Direct\n";
    
    fft_opencl_c2c fo({n_x[0]});
 
    sw.start();
    for(i=0; i<n_coils; ++i){
      fo.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";    

    std::cout << "        Inverse\n";

    fft_opencl_c2c foi({n_x[0]}, true);
   
    sw.start();
    for(i=0; i<n_coils; ++i){
      foi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "        Error: " << error << "\n";
    std::cout << "\n\n";

  }
  
  
  // 2D
  n_dimensions=2;
  n_x[0]=dim[0];
  n_x[1]=dim[1];
  
  
  // R <-> HC
  
  {
    // Dimensions
    
    std::cout << "Dimensions: ";
    for(i=0; i<n_dimensions; ++i){
      std::cout << n_x[i] << " ";
    }
    std::cout << "\n";
    std::cout << "N coils: " << n_coils << "\n";
    std::cout << "\n";
    std::cout << "  R <-> HC\n";
    std::cout << "\n";
    
    // Signal
    multiarray<double> sig({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];  
        s=signal(n_dimensions, l_x, a, b, x);
        sig(k[0], k[1])=s;
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        coil(k[0], k[1])=1./n_coils;
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    
    fft_opencl_r2c fo({n_x[0], n_x[1]});
    
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]})); 
      inverse_transforms.push_back(multiarray<double>({n_x[0], n_x[1]}));
    }
   
    std::cout << "    FFT\n";
    // clFFT
    std::cout << "      clFFT\n";
    std::cout << "        Direct\n";

    
    sw.start();
    for(i=0; i<n_coils; ++i){
      fo.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";    

    std::cout << "        Inverse\n";

    fft_opencl_r2c foi({n_x[0], n_x[1]}, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      foi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "        Error: " << error << "\n";
    std::cout << "\n";
    
  }  
  
  
  // C <-> C
  
  {
    std::cout << "  C <-> C\n";
    std::cout << "\n";
    
    // Signal
    multiarray<::complex> sig({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];  
        s=signal(n_dimensions, l_x, a, b, x);
        sig(k[0], k[1])=::complex(s, s);
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        coil(k[0], k[1])=1./n_coils;
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<::complex>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<::complex>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(multiarray<::complex>({n_x[0], n_x[1]})); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]}));
    }
        int n_queues=1;
    std::cout << "      OpenCL (" << n_queues << " queues)\n";
    opencl_product_c opc1(sig.size(), n_queues);
    sw.start();
    for(i=0; i<n_coils; ++i){
      opc1.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    n_queues=4;
    std::cout << "      OpenCL (" << n_queues << " queues)\n";
    opencl_product_c opc(sig.size(), n_queues);
    sw.start();
    for(i=0; i<n_coils; ++i){
      opc.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    std::cout << "    FFT\n";
    // CUDA
    std::cout << "      clFFT\n";
    std::cout << "        Direct\n";

    fft_opencl_c2c fo({n_x[0], n_x[1]});
    sw.start();
    for(i=0; i<n_coils; ++i){
      fo.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";    

    std::cout << "        Inverse\n";

    fft_opencl_c2c foi({n_x[0], n_x[1]}, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      foi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "        Error: " << error << "\n";
    std::cout << "\n\n";

  }  
  
  

  // 3D
  n_dimensions=3;
  n_x[0]=dim[0];
  n_x[1]=dim[1];
  n_x[2]=dim[2];
  
  
  // R <-> HC
  
  {
    // Dimensions
    
    std::cout << "Dimensions: ";
    for(i=0; i<n_dimensions; ++i){
      std::cout << n_x[i] << " ";
    }
    std::cout << "\n";
    std::cout << "N coils: " << n_coils << "\n";
    std::cout << "\n";
    std::cout << "  R <-> HC\n";
    std::cout << "\n";
    
    // Signal
    multiarray<double> sig({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){
          x[2]=((double)k[2])/n_x[2];
          s=signal(n_dimensions, l_x, a, b, x);
          sig(k[0], k[1], k[2])=s;
        }
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){   
          coil(k[0], k[1], k[2])=1./n_coils;
        }
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1], n_x[2]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0], n_x[1], n_x[2]}));
    }
    
     std::cout << "    FFT\n";
    // OpenCL
    std::cout << "      OpenCL\n";
    std::cout << "        Direct\n";

    fft_opencl_r2c fo({n_x[0], n_x[1], n_x[2]});
    sw.start();
    for(i=0; i<n_coils; ++i){
      fo.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";    

    std::cout << "        Inverse\n";

    fft_opencl_r2c foi({n_x[0], n_x[1], n_x[2]}, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      foi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "        Error: " << error << "\n";
    std::cout << "\n";
  }  
  
  
  // C <-> C
  
  {
    std::cout << "  C <-> C\n";
    std::cout << "\n";
    
    // Signal
    multiarray<::complex> sig({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      x[0]=((double)k[0])/n_x[0];
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){
        x[1]=((double)k[1])/n_x[1];
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){
          x[2]=((double)k[2])/n_x[2];
          s=signal(n_dimensions, l_x, a, b, x);
          sig(k[0], k[1], k[2])=::complex(s, s);
        }
      }
    }
    
    // Coils
    multiarray<double> coil({n_x[0], n_x[1], n_x[2]});
    for(k[0]=0; k[0]<n_x[0]; ++k[0]){
      for(k[1]=0; k[1]<n_x[1]; ++k[1]){  
        for(k[2]=0; k[2]<n_x[2]; ++k[2]){   
          coil(k[0], k[1], k[2])=1./n_coils;
        }
      }
    }   
    std::vector<multiarray<double>> coils;
    for(i=0; i<n_coils; ++i){
      coils.push_back(coil);   
    }
    
    // Multiplied signals
    std::vector<multiarray<::complex>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<::complex>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1], n_x[2]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0], n_x[1], n_x[2]}));
    }

        std::cout << "    FFT\n";
    // OpenCL
    std::cout << "      OpenCL\n";
    std::cout << "        Direct\n";

    fft_opencl_c2c fo({n_x[0], n_x[1], n_x[2]});
    sw.start();
    for(i=0; i<n_coils; ++i){
      fo.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";    

    std::cout << "        Inverse\n";

    fft_opencl_c2c foi({n_x[0], n_x[1], n_x[2]}, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      foi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "          Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "        Error: " << error << "\n";
    std::cout << "\n\n";

  }  
    
  //fftw_cleanup_threads();
  return 0;

}
