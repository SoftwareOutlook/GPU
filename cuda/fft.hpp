#ifndef FFTW_HPP
#define FFTW_HPP

#include "complex.hpp"
#include <fftw3.h>
#include <math.h>


   
class fft{
public:
  typedef int size_t;
protected:
  size_t n_dimensions;
  size_t* n;
  bool inverse;

public:
  fft(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse){
    n_dimensions=i_n_dimensions;
    n=new size_t[n_dimensions];
    for(size_t i=0; i<n_dimensions; ++i){
      n[i]=i_n[i];
    }
    inverse=i_inverse;
  }
  ~fft(){
    delete[] n;
  }
  size_t get_n_dimensions() const {
    return n_dimensions;
  }
  size_t size(const size_t i) const {
    return n[i];
  }
  size_t size() const {
    size_t s=1;
    for(size_t i=0; i<n_dimensions; ++i){
      s=s*n[i];
    }
    return s;
  }
  bool is_inverse() const {
    return inverse;   
  }
  virtual size_t size_complex() const = 0;
  virtual int compute(::complex* in, ::complex* out) const {}
  virtual int compute(double* in, ::complex* out) const {}
};


class fftw : public fft {
protected:
  fftw_plan plan;
public:
  fftw(const size_t i_n_dimensions, const size_t* i_n, bool i_inverse) : fft(i_n_dimensions, i_n, i_inverse){
  }
};


class fftw_c2c : public fftw{
private:
  mutable fftw_complex *fftw_in, *fftw_out;
public:
  fftw_c2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : fftw(i_n_dimensions, i_n_x, i_inverse){
    fftw_in=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size());
    fftw_out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size());
    if(!is_inverse()){
      plan=fftw_plan_dft(n_dimensions, n, fftw_in, fftw_out,FFTW_FORWARD, FFTW_ESTIMATE);
    }else{
      plan=fftw_plan_dft(n_dimensions, n, fftw_in, fftw_out,FFTW_BACKWARD, FFTW_ESTIMATE);  
    }
  }
  ~fftw_c2c(){
    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
  }
  size_t size_complex() const {
    return size();
  }
  int compute(::complex* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=0; i<size(); ++i){
        fftw_in[i][0]=in[i].real();
        fftw_in[i][1]=in[i].imag();
      }
      fftw_execute(plan);
      for(i=0; i<size(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=0; i<size(); ++i){
        fftw_in[i][0]=out[i].real();
        fftw_in[i][1]=out[i].imag();
      }
      fftw_execute(plan);
      for(i=0; i<size(); ++i){
        in[i].x=fftw_out[i][0]/size();
        in[i].y=fftw_out[i][1]/size();
      }       
    }
    return 0;
  }
};


class fftw_r2c : public fftw{
private:
  mutable double* fftw_in;
  mutable fftw_complex* fftw_out;
public:
  fftw_r2c(const size_t i_n_dimensions, const size_t* i_n_x, bool i_inverse=false) : fftw(i_n_dimensions, i_n_x, i_inverse){
    fftw_in=(double*)fftw_malloc(sizeof(double)*size());
    fftw_out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*size());
    if(!is_inverse()){
      plan=fftw_plan_dft_r2c(n_dimensions, n, fftw_in, fftw_out, FFTW_ESTIMATE);
    }else{
      plan=fftw_plan_dft_c2r(n_dimensions, n, fftw_out, fftw_in, FFTW_ESTIMATE);  
    }
  }
  ~fftw_r2c(){
    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
  }
  size_t size_complex() const {
    size_t s=1;
    for(size_t i=0; i<n_dimensions-1; ++i){
      s=s*n[i];
    }
    s=s*(n[n_dimensions-1]/2+1);
    return s;
  }
  int compute(double* in, ::complex* out) const {
    size_t i;
    if(!is_inverse()){
      for(i=0; i<size(); ++i){
        fftw_in[i]=in[i];
      }
      fftw_execute(plan);
      for(i=0; i<size_complex(); ++i){
        out[i].x=fftw_out[i][0];
        out[i].y=fftw_out[i][1];
      }
    }else{
      for(i=0; i<size_complex(); ++i){
        fftw_out[i][0]=out[i].x;
        fftw_out[i][1]=out[i].y;
      }
      fftw_execute(plan);
      for(i=0; i<size(); ++i){
        in[i]=fftw_in[i]/size();
      }
    }
    return 0;
  }
};

#endif
