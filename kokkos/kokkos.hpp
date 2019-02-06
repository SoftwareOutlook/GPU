#ifndef KOKKOS_HPP
#define KOKKOS_HPP

#include <multiarray.hpp>
#include <complex>

void kokkos_loop(const int n, double* z_re, double* z_im, double* a_val, double* az_re, double* az_im);

void kokkos_loop(const int n, double* x, double* a, double* ax);

int kokkos_product(const multiarray<::complex>& z, const multiarray<double>& a, multiarray<::complex>& az);

int kokkos_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax);


#endif
