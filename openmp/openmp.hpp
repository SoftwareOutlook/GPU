#ifndef OPENMP_HPP
#define OPENMP_HPP

#include <multiarray.hpp>
#include <omp.h>

template <class T, class U, class V> int openmp_product_cpu(const multiarray<T>& a, const multiarray<U>& b,  multiarray<V>& ab){
#pragma omp parallel for
  for(typename multiarray<T>::size_t i; i<a.size(); ++i){
    ab.set(i, a.get(i)*b.get(i));
  }
  return 0;
}

template<class T, class U, class V> void openmp_product_gpu(const multiarray<T>& a, const multiarray<U>& b,  multiarray<V>& ab){
#pragma omp target map(to:a, b) map(from:ab)
#pragma omp teams distribute parallel for simd
  for(size_t i=0; i<a.size(); ++i){
    ab.set(i, a.get(i)*b.get(i));
  }
}


#endif
