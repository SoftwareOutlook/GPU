#ifndef MULTIARRAY_HPP
#define MULTIARRAY_HPP

#include <vector>
#include "complex"


template<class T> class multiarray{
public:
  typedef long size_t;
private:
  std::vector<size_t> dimensions;
  T* t;
  inline size_t compute_index(const size_t* indices) const {
    size_t i, index=0, stride=1;
    for(i=n_dimensions()-1; i>=0; --i){
      index=index+stride*indices[i];
      stride=stride*size(i);
    }
    return index;
  }
public:
  multiarray(const std::vector<size_t>& i_dimensions){
    dimensions=i_dimensions;
    t=new T[size()];
  }
  multiarray(const multiarray& other) : multiarray(other.dimensions) {
    for(size_t i=0; i<size(); ++i){
      t[i]=other.t[i];
    }
  }
  multiarray(multiarray&& other){
    dimensions=other.dimensions;
    t=other.t;
    other.t=nullptr;
  }
  multiarray operator=(const multiarray& other){
    if(this!=&other){
      dimensions=other.dimensions;
      for(size_t i=0; i<size(); ++i){
        t[i]=other.t[i];
      }
    }
    return *this;
  }
  multiarray operator=(multiarray&& other){
    if(this!=&other){
      dimensions=other.dimensions;
      delete[] t;
      t=other.t;
      other.t=nullptr;
    }
    return *this;
  }
  ~multiarray(){
    delete[] t;
  }
  inline size_t n_dimensions() const {
    return dimensions.size();
  }
  inline size_t size(const size_t i) const {
    return dimensions[i];
  }
  inline size_t size() const {
    size_t i, n=1;
    for(i=0; i<n_dimensions(); ++i){
      n=n*dimensions[i];
    }
    return n;
  }
  /*
  template<class ...U> inline T& operator()(U ...indices){
    size_t i=0, k[sizeof...(indices)];
    (... , void(k[i++] = indices));
    return t[compute_index(k)];
  }
  */
  inline T& operator()(const size_t* indices){
    return t[compute_index(indices)];
  }
  inline T& operator()(const size_t i_x){
    return t[compute_index(&i_x)];
  }
  inline T& operator()(const size_t i_x, const size_t i_y){
    size_t indices[2];
    indices[0]=i_x;
    indices[1]=i_y;
    return t[compute_index(indices)];
  }
  inline T& operator()(const size_t i_x, const size_t i_y, const size_t i_z){
    size_t indices[3];
    indices[0]=i_x;
    indices[1]=i_y;
    indices[2]=i_z;
    return t[compute_index(indices)];
  }
  inline T get(const size_t i) const {
    return t[i];   
  }
  inline multiarray operator+(const multiarray& other){
    multiarray result(dimensions);
    for(size_t i=0; i<size(); ++i){
      result.t[i]=t[i]+other.t[i];
    }
    return result;
  }
  inline multiarray operator-(const multiarray& other){
    multiarray result(dimensions);
    for(size_t i=0; i<size(); ++i){
      result.t[i]=t[i]-other.t[i];
    }
    return result;
  }
  template<class U> inline multiarray operator*(const multiarray<U>& other){
    multiarray result(dimensions);
    for(size_t i=0; i<size(); ++i){
      result.t[i]=t[i]*other.get(i);
    }
    return result;
  }
  inline multiarray operator*(const double a){
    multiarray result(dimensions);
    for(size_t i=0; i<size(); ++i){
      result.t[i]=t[i]*a;
    }
    return result;
  }
  template<class U> inline multiarray operator/(const multiarray<U>& other){
    multiarray result(dimensions);
    for(size_t i=0; i<size(); ++i){
      result.t[i]=t[i]/other.get[i];
    }
    return result;
  }
  inline multiarray operator/(const double a){
    multiarray result(dimensions);
    for(size_t i=0; i<size(); ++i){
      result.t[i]=t[i]/a;
    }
    return result;
  }
  double norm() const{
    double N=0;
    for(size_t i=0; i<size(); ++i){
      N=N+::norm(t[i]);   
    }
    return N;
  }
  inline T* pointer() const {
    return t;   
  }
};


#endif
