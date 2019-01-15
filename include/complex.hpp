#ifndef COMPLEX_HPP
#define COMPLEX_HPP

#include <math.h>
#include <iostream>

    
class complex{
public:
  typedef double float_t;
  float_t x, y;
  complex(const float_t& i_x=0, const float_t& i_y=0){
    x=i_x;
    y=i_y;
  }
  inline float_t real() const {
    return x;   
  }
  inline float_t imag() const {
    return y;   
  }
  inline float_t norm() const {
    return sqrt(x*x+y*y);
  }
  inline complex operator+(const complex& z){
    return complex(x+z.x, y+z.y);    
  }
  inline complex operator-(const complex& z){
    return complex(x-z.x, y-z.y);    
  }
  inline complex operator*(const complex& z){
    return complex(x*z.x-y*z.y, x*z.y+y*z.x);    
  }
  inline complex operator*(const float_t& a){
    return complex(a*x, a*y);    
  }
  inline complex operator/(const float_t& a){
    return complex(x/a, y/a);    
  }
  inline friend std::ostream& operator<<(std::ostream &out, const complex& z){
    out << z.real() << " " << z.imag();
    return out;
  }

};

double norm(const double& a){
  if(a>=0){
    return a;   
  }else{
    return -a;   
  }
}

double norm(const complex& z){
  return sqrt(pow(z.real(), 2)+pow(z.imag(), 2));   
}


#endif
