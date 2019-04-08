#include "complex.hpp"
#include "signal.hpp"
#include <iostream>
#include "stopwatch.hpp"
#include "multiarray.hpp"
#include <vector>
#include <omp.h>
#include "openmp.hpp"

using namespace std;


int main(int argc, char** argv){

    
  if(argc<3){
    std::cout << "Please supply a dimension and a number of coils\n";
    return -1;
  }

  
  int i, j, k[3], n_x[3], dim, n_dimensions, n_coils;
  double a=0.25;
  double b=a/2;
  double s, x[3], l_x[3]={1, 1, 1}, error;
  dim=atoi(*(argv+1));
  n_coils=atoi(*(argv+2));
  
  stopwatch sw;


  // 1D
  
  n_dimensions=1;
  n_x[0]=dim;
  
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
      transforms.push_back(multiarray<::complex>({n_x[0]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0]}));
    }

    std::cout << "    Multiplication\n";

    // CPU
    std::cout << "      CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      multiplied_signals[i]=sig*coils[i];
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    // OpenMP CPU
    std::cout << "      OpenMP CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_cpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    // OpenMP GPU
    std::cout << "      OpenMP GPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_gpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
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
      multiplied_signals.push_back(multiarray<::complex>({n_x[0]}));
      transforms.push_back(multiarray<::complex>({n_x[0]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0]}));
    }

    std::cout << "    Multiplication\n";

    // CPU
    std::cout << "      CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      multiplied_signals[i]=sig*coils[i];
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";
 
    // OpenMP
    std::cout << "      OpenMP\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_cpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    // OpenMP GPU
    std::cout << "      OpenMP GPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_gpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";
  }



  // 2D
  
  n_dimensions=2;
  n_x[0]=dim;
  n_x[1]=dim;
  
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
   
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      // multiplied_signals.push_back(sig*coils[i]);
      multiplied_signals.push_back(multiarray<double>({n_x[0], n_x[1]}));
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0], n_x[1]}));
    }

    std::cout << "    Multiplication\n";

    // CPU
    std::cout << "      CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      multiplied_signals[i]=sig*coils[i];
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    // OpenMP CPU
    std::cout << "      OpenMP CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_cpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    // OpenMP GPU
    std::cout << "      OpenMP GPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_gpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

  }

  // C <-> C
  
  {
    // Dimensions
    
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

    std::cout << "    Multiplication\n";

    // CPU
    std::cout << "      CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      multiplied_signals[i]=sig*coils[i];
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";
 
    // OpenMP
    std::cout << "      OpenMP\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_cpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";

    // OpenMP GPU
    std::cout << "      OpenMP GPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      openmp_product_gpu(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    std::cout << "\n";
  }

  
  
  return 0;

}
