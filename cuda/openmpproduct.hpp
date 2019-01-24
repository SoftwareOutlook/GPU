#ifndef OPENMPPRODUCT_HPP
#define OPENMPPRODUCT_HPP

#include <multiarray.hpp>
#include <omp.h>

template <class T, class U, class V> int openmp_product(const multiarray<T>& a, const multiarray<U>& b,  multiarray<V>& ab){
#pragma omp parallel for
  for(typename multiarray<T>::size_t i; i<a.size(); ++i){
    ab.set(i, a.get(i)*b.get(i));
  }
  return 0;
}


#endif
