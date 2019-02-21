#include "complex.hpp"
#include "signal.hpp"
#include <iostream>
#include "stopwatch.hpp"
#include "multiarray.hpp"
#include <vector>
#include <omp.h>
#include "kokkos.hpp"
#include <Kokkos_Core.hpp>
using namespace std;


int main(int argc, char** argv){

  if(argc<3){
    std::cout << "Please supply a dimension and a number of coils\n";
    return -1;
  }
  Kokkos::initialize(argc, argv);
 
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
  
  // Real
  
  {
    // Dimensions
    
    std::cout << "Dimensions: ";
    for(i=0; i<n_dimensions; ++i){
      std::cout << n_x[i] << " ";
    }
    std::cout << "\n";
    std::cout << "N coils: " << n_coils << "\n";
    std::cout << "\n";
    std::cout << "  Real\n";
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
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(multiarray<double>({n_x[0]}));
    }

    std::cout << "    Multiplication\n";

    // CPU
    std::cout << "      CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      multiplied_signals[i]=sig*coils[i];
    }
    sw.stop();
    
    std::cout << "        Time:   " << sw.get() << " s\n";
    std::cout << "\n";

    // Kokkos (multithreading)
    product_kokkos_real kp(sig.size());
    std::cout << "      Kokkos (multithreading)\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      kp.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:   " << sw.get() << " s\n";
    error=(multiplied_signals.front()-sig/n_coils).norm();
    std::cout << "        Error:  " << error << "\n";
    std::cout << "\n";
    
    // Kokkos (GPU)
    std::cout << "      Kokkos (GPU)\n";
    Kokkos::View<double*, Kokkos::HostSpace> h_z("h_z", sig.size()); // must be in same scope
    Kokkos::View<double*, Kokkos::HostSpace> h_a("h_a", sig.size());
    Kokkos::View<double*, Kokkos::HostSpace> h_az("h_az", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_z("d_z", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_a("d_a", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_az("d_az", sig.size());
    sw.start();
    for(i=0; i<n_coils; ++i){
      kokkos_product_gpu(sig, coils[i], multiplied_signals[i], h_z, h_a, h_az, d_z, d_a, d_az);
    }
    sw.stop();
    std::cout << "        Time:   " << sw.get() << " s\n";
    error=(multiplied_signals.front()-sig/n_coils).norm();
    std::cout << "        Error:  " << error << "\n";
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
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(multiarray<::complex>({n_x[0]}));
    }

    std::cout << "    Multiplication\n";

    // CPU
    std::cout << "      CPU\n";
    sw.start();
    for(i=0; i<n_coils; ++i){
      multiplied_signals[i]=sig*coils[i];
    }
    sw.stop();
    std::cout << "        Time:   " << sw.get() << " s\n";
    std::cout << "\n";
 
    // Kokkos (multithreading)
    std::cout << "      Kokkos (multithreading)\n";
    product_kokkos_complex kp(sig.size());
    sw.start();
    for(i=0; i<n_coils; ++i){
      kp.compute(sig, coils[i], multiplied_signals[i]);
    }
    sw.stop();
    std::cout << "        Time:   " << sw.get() << " s\n";
    error=(multiplied_signals.front()-sig/n_coils).norm();
    std::cout << "        Error:  " << error << "\n";
    std::cout << "\n";

    // Kokkos (GPU)
    std::cout << "      Kokkos (GPU)\n";
    Kokkos::View<double*, Kokkos::HostSpace> h_z_re("h_z_re", sig.size());
    Kokkos::View<double*, Kokkos::HostSpace> h_z_im("h_z_im", sig.size());
    Kokkos::View<double*, Kokkos::HostSpace> h_a("h_a", sig.size());
    Kokkos::View<double*, Kokkos::HostSpace> h_az_re("h_az_re", sig.size());
    Kokkos::View<double*, Kokkos::HostSpace> h_az_im("h_az_im", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_z_re("d_z_re", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_z_im("d_z_im", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_a("d_a", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_az_re("d_az_re", sig.size());
    Kokkos::View<double*, Kokkos::CudaSpace> d_az_im("d_az_im", sig.size());
    sw.start();
    for(i=0; i<n_coils; ++i){
      kokkos_product_gpu(sig, coils[i], multiplied_signals[i], h_z_re, h_z_im, h_a, h_az_re, h_az_im, d_z_re, d_z_im, d_a, d_az_re, d_az_im);
    }
    sw.stop();
    std::cout << "        Time:   " << sw.get() << " s\n";
    error=(multiplied_signals.front()-sig/n_coils).norm();
    std::cout << "        Error:  " << error << "\n";
    std::cout << "\n";
  }
 
  Kokkos::finalize();
  return 0;

}
 
