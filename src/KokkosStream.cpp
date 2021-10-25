// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "KokkosStream.hpp"

template <class T>
KokkosStream<T>::KokkosStream(
        const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE),
      d_a("d_a", ARRAY_SIZE),
      d_b("d_b", ARRAY_SIZE),
      d_c("d_c", ARRAY_SIZE),
      hm_a(create_mirror_view(d_a)),
      hm_b(create_mirror_view(d_b)),
      hm_c(create_mirror_view(d_c))
{
}

template <class T>
KokkosStream<T>::~KokkosStream()
{
}

template <class T>
void KokkosStream<T>::init_arrays(T initA, T initB, T initC)
{
  Kokkos::parallel_for(array_size, KOKKOS_CLASS_LAMBDA (const long index)
  {
    d_a[index] = initA;
    d_b[index] = initB;
    d_c[index] = initC;
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::read_arrays(std::vector<T>& a,
                                  std::vector<T>& b,
                                  std::vector<T>& c)
{
  deep_copy(hm_a, d_a);
  deep_copy(hm_b, d_b);
  deep_copy(hm_c, d_c);
  for(int ii = 0; ii < array_size; ++ii)
  {
    a[ii] = (hm_a)(ii);
    b[ii] = (hm_b)(ii);
    c[ii] = (hm_c)(ii);
  }
}

template <class T>
void KokkosStream<T>::copy()
{
  Kokkos::parallel_for(array_size, KOKKOS_CLASS_LAMBDA (const long index)
  {
    d_c[index] = d_a[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::mul()
{
  const T scalar = startScalar;
  Kokkos::parallel_for(array_size, KOKKOS_CLASS_LAMBDA (const long index)
  {
    d_b[index] = scalar * d_c[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::add()
{
  Kokkos::parallel_for(array_size, KOKKOS_CLASS_LAMBDA (const long index)
  {
    d_c[index] = d_a[index] + d_b[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::triad()
{

  const T scalar = startScalar;
  Kokkos::parallel_for(array_size, KOKKOS_CLASS_LAMBDA (const long index)
  {
    d_a[index] = d_b[index] + scalar * d_c[index];
  });
  Kokkos::fence();
}

template <class T>
T KokkosStream<T>::dot()
{

  T sum = 0.0;

  Kokkos::parallel_reduce(array_size, KOKKOS_CLASS_LAMBDA (const long index, T &tmp)
  {
    tmp += d_a[index] * d_b[index];
  }, sum);

  return sum;

}

void listDevices(void)
{
  std::cout << "Kokkos library for " << getDeviceName(0) << std::endl;
}


std::string getDeviceName(const int device)
{
  return typeid (Kokkos::DefaultExecutionSpace).name();
}


std::string getDeviceDriver(const int device)
{
  return "Kokkos";
}

template class KokkosStream<float>;
template class KokkosStream<double>;
