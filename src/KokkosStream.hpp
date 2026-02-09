// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "Stream.h"

// #define IMPLEMENTATION_STRING "Kokkos"

namespace impl
{

template <typename T>
void
copy(Kokkos::View<T *> const & d_a, Kokkos::View<T *> const & d_c)
{
  Kokkos::parallel_for(
    "copy",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, d_a.extent(0)),
    KOKKOS_LAMBDA(const long index) { d_c[index] = d_a[index]; });
  Kokkos::fence();
}

template <class T>
void
add(Kokkos::View<T *> const & d_a, Kokkos::View<T *> const & d_b, Kokkos::View<T *> const & d_c)
{
  Kokkos::parallel_for(
    "add",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, d_a.extent(0)),
    KOKKOS_LAMBDA(const long index) { d_c[index] = d_a[index] + d_b[index]; });
  Kokkos::fence();
}

template <class T>
void
mul(Kokkos::View<T *> const & d_b, Kokkos::View<T *> const & d_c)
{
  const T scalar = startScalar;
  Kokkos::parallel_for(
    "mul",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, d_b.extent(0)),
    KOKKOS_LAMBDA(const long index) { d_b[index] = scalar * d_c[index]; });
  Kokkos::fence();
}

template <class T>
void
triad(Kokkos::View<T *> const & d_a, Kokkos::View<T *> const & d_b, Kokkos::View<T *> const & d_c)
{

  const T scalar = startScalar;
  Kokkos::parallel_for(
    "triad",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, d_a.extent(0)),
    KOKKOS_LAMBDA(const long index) { d_a[index] = d_b[index] + scalar * d_c[index]; });
  Kokkos::fence();
}

template <class T>
T
dot(Kokkos::View<T *> const & d_a, Kokkos::View<T *> const & d_b)
{

  T sum = 0.0;

  Kokkos::parallel_reduce(
    "dot",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, d_a.extent(0)),
    KOKKOS_LAMBDA(const long index, T & tmp) { tmp += d_a[index] * d_b[index]; },
    Kokkos::Sum<T>(sum));

  return sum;
}

} // namespace impl

template <class T>
class KokkosStream : public Stream<T>
{
protected:
  // Size of arrays
  unsigned int array_size;

  using view_t = Kokkos::View<T *>;
  using mirror_view_t = typename view_t::host_mirror_type;

  // Device side pointers to arrays
  view_t        d_a;
  view_t        d_b;
  view_t        d_c;
  mirror_view_t hm_a;
  mirror_view_t hm_b;
  mirror_view_t hm_c;

public:
  KokkosStream(const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
    , d_a("d_a", ARRAY_SIZE)
    , d_b("d_b", ARRAY_SIZE)
    , d_c("d_c", ARRAY_SIZE)
    , hm_a(create_mirror_view(d_a))
    , hm_b(create_mirror_view(d_b))
    , hm_c(create_mirror_view(d_c))
  {}

  ~KokkosStream() = default;

  virtual void
  copy() override
  {
    impl::copy(d_a, d_c);
  }

  virtual void
  add() override
  {
    impl::add(d_a, d_b, d_c);
  }

  virtual void
  mul() override
  {
    impl::mul(d_b, d_c);
  }

  virtual void
  triad() override
  {
    impl::triad(d_a, d_b, d_c);
  }

  virtual T
  dot() override
  {
    return impl::dot(d_a, d_b);
  }

  virtual void
  init_arrays(T initA, T initB, T initC) override
  {
    Kokkos::deep_copy(d_a, initA);
    Kokkos::deep_copy(d_b, initB);
    Kokkos::deep_copy(d_c, initC);
  }

  virtual void
  read_arrays(std::vector<T> & a, std::vector<T> & b, std::vector<T> & c) override
  {
    deep_copy(hm_a, d_a);
    deep_copy(hm_b, d_b);
    deep_copy(hm_c, d_c);
    for (int ii = 0; ii < array_size; ++ii)
    {
      a[ii] = (hm_a)(ii);
      b[ii] = (hm_b)(ii);
      c[ii] = (hm_c)(ii);
    }
  }

}; // class KokkosStream

extern template class KokkosStream<float>;
extern template class KokkosStream<double>;
