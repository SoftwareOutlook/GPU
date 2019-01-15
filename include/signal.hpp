#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include "complex.hpp"
#include <vector>
#include <math.h>
using namespace std;


double signal(const unsigned long n_dimensions, const double* l_x, const double a, const double b, const double* x){
  unsigned long i;
  double lower_bound[n_dimensions], upper_bound[n_dimensions];
  for(i=0; i<n_dimensions; ++i){
    lower_bound[i]=a*l_x[i];
    upper_bound[i]=(1-b)*l_x[i];
  }
  for(i=0; i<n_dimensions; ++i){
    if(x[i]>=lower_bound[i] && x[i]<upper_bound[i]){
      return 0;   
    }
  }
  return 1;
}

double signal1d(const double l_x, const double a, const double b, const double x){
  return signal(1, &l_x, a, b, &x);   
}

double signal2d(const double i_l_x, const double i_l_y, const double a, const double b, const double i_x, const double i_y){
  double l_x[2]={i_l_x, i_l_y};
  double x[2]={i_x, i_y};
  return signal(2, l_x, a, b, x);
}

double signal3d(const double i_l_x, const double i_l_y, const double i_l_z, const double a, const double b, const double i_x, const double i_y, const double i_z){
  double l_x[3]={i_l_x, i_l_y, i_l_z};
  double x[3]={i_x, i_y, i_z};
  return signal(3, l_x, a, b, x);
}


#endif
