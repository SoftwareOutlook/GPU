#include "complex.hpp"
#include "signal.hpp"
#include <iostream>
#include "stopwatch.hpp"
#include "multiarray.hpp"
#include <vector>
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
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0]}));
    }
    



    /*
    // FFTW
    std::cout << "    FFTW\n";
    std::cout << "      Direct\n";
    
    fftw_r2c fw(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftw_r2c fwi(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    
    
    
    std::cout << "    GSL\n";
    std::cout << "      Direct\n";
    
    gsl_fft_r2c gsl(n_x[0]);

    sw.start();
    for(i=0; i<n_coils; ++i){
      gsl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
     
    std::cout << "      Inverse\n";
    
    gsl_fft_r2c gsli(n_x[0], true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      gsli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";

    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    
    
    
    std::cout << "    MKL\n";
    std::cout << "      Direct\n";
    
    mkl_fft_r2c mkl(n_dimensions, n_x);
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    std::cout << "      Inverse\n";
    mkl_fft_r2c mkli(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
 
    
    std::cout << "    FFTPACK\n";
    std::cout << "      Direct\n";
    
    fftpack_r2c fftpack(n_x[0]);
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      fftpack.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    std::cout << "      Inverse\n";
    fftpack_r2c fftpacki(n_x[0]);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fftpacki.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    */
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
    
    



    /*
    // FFTW
    std::cout << "    FFTW\n";
    std::cout << "      Direct\n";
    
    fftw_c2c fw(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftw_c2c fwi(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n"; 
    
    
    // GSL
    std::cout << "    GSL\n";
    std::cout << "      Direct\n";
    
    
    gsl_fft_c2c gsl(n_x[0]);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    gsl_fft_c2c gsli(n_x[0], true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      gsli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n"; 
    
    
    // MKL
    std::cout << "    MKL\n";
    std::cout << "      Direct\n";
    
    
    mkl_fft_c2c mkl(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    mkl_fft_c2c mkli(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      gsli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";    
    
    
    // FFTPACK
    std::cout << "    FFTPACK\n";
    std::cout << "      Direct\n";
    
    fftpack_c2c fftpack(n_x[0]);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftpack_c2c fftpacki(n_x[0], true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      gsli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";    
    std::cout << "\n"; 
    */    
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
    
    // Multiplied signals
    std::vector<multiarray<double>> multiplied_signals;
    std::vector<multiarray<::complex>> transforms;
    std::vector<multiarray<double>> inverse_transforms;
    for(i=0; i<n_coils; ++i){
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]/2+1}));
      inverse_transforms.push_back(multiarray<double>({n_x[0], n_x[1]}));
    }
    
    



    /*

    // FFTW
    std::cout << "    FFTW\n";
    std::cout << "      Direct\n";
    
    fftw_r2c fw(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftw_r2c fwi(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    
    
    // MKL
    std::cout << "    MKL\n";
    std::cout << "      Direct\n";
    
    mkl_fft_r2c mkl(n_dimensions, n_x);
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    std::cout << "      Inverse\n";
    mkl_fft_r2c mkli(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
*/    
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
      multiplied_signals.push_back(sig*coils[i]); 
      transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]}));
      inverse_transforms.push_back(multiarray<::complex>({n_x[0], n_x[1]}));
    }
    
    /*
    // FFTW
    std::cout << "    FFTW\n";
    std::cout << "      Direct\n";
    
    fftw_c2c fw(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftw_c2c fwi(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    
    
    // MKL
    std::cout << "    MKL\n";
    std::cout << "      Direct\n";
    
    mkl_fft_c2c mkl(n_dimensions, n_x);
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    std::cout << "      Inverse\n";
    mkl_fft_c2c mkli(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    */
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
    
    /*
    // FFTW
    std::cout << "    FFTW\n";
    std::cout << "      Direct\n";
    
    fftw_r2c fw(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftw_r2c fwi(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    
    
    // MKL
    std::cout << "    MKL\n";
    std::cout << "      Direct\n";
    
    mkl_fft_r2c mkl(n_dimensions, n_x);
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    std::cout << "      Inverse\n";
    mkl_fft_r2c mkli(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    */
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
    
    /*
    // FFTW
    std::cout << "    FFTW\n";
    std::cout << "      Direct\n";
    
    fftw_c2c fw(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fw.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    
    std::cout << "      Inverse\n";
    
    fftw_c2c fwi(n_dimensions, n_x, true);
    sw.start();
    for(i=0; i<n_coils; ++i){
      fwi.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    
    
    // MKL
    std::cout << "    MKL\n";
    std::cout << "      Direct\n";
    
    mkl_fft_c2c mkl(n_dimensions, n_x);
    
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkl.compute(multiplied_signals[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    std::cout << "      Inverse\n";
    mkl_fft_c2c mkli(n_dimensions, n_x);
    sw.start();
    for(i=0; i<n_coils; ++i){
      mkli.compute(inverse_transforms[i].pointer(), transforms[i].pointer());
    }
    sw.stop();
    std::cout << "        Time:  " << sw.get() << " s\n";
    
    error=0;
    for(i=0; i<n_coils; ++i){
      error=error+(multiplied_signals[i]-inverse_transforms[i]).norm();
    }
    std::cout << "      Error: " << error << "\n";
    std::cout << "\n";
    */
  }  
   
   
  //fftw_cleanup_threads();

  return 0;

}
