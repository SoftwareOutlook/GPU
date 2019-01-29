#ifndef OPENACCPRODUCT_HPP
#define OPENACCPRODUCT_HPP

#include <complex.hpp>
#include <multiarray.hpp>

int openacc_product(const multiarray<complex>& z, const multiarray<double>& a, multiarray<complex>& az);

int openacc_product(const multiarray<double>& x, const multiarray<double>& a, multiarray<double>& ax);


#endif
