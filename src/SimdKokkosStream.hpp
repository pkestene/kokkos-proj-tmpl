// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <simd.hpp>

#include "Stream.h"

//#define IMPLEMENTATION_STRING "SimdKokkos"

// ====================================================================
// ====================================================================
template <class T>
class SimdKokkosStream : public Stream<T>
{
public:
  // Size of arrays
  unsigned int array_size_scalar;
  unsigned int array_size_vector;

#ifdef KOKKOS_ENABLE_CUDA
  using simd_t = typename simd::simd<T,simd::simd_abi::cuda_warp<32>>;
#else
  using simd_t = typename simd::simd<T,simd::simd_abi::native>;
  //using simd_t = typename simd::simd<T,simd::simd_abi::pack<8>>;
#endif
  using simd_storage_t = typename simd_t::storage_type;

  using view_t = Kokkos::View<T*>;
  using mirror_view_t = typename view_t::HostMirror;

protected:
  // Device arrays
  view_t d_a;
  view_t d_b;
  view_t d_c;

  // Host mirrors
  mirror_view_t hm_a;
  mirror_view_t hm_b;
  mirror_view_t hm_c;

public:

  SimdKokkosStream(const unsigned int, const int);
  ~SimdKokkosStream();

  virtual void copy() override;
  virtual void add() override;
  virtual void mul() override;
  virtual void triad() override;
  virtual T dot() override;

  virtual void init_arrays(T initA, T initB, T initC) override;
  virtual void read_arrays(
     std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

}; // class SimdKokkosStream

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
template<>
struct reduction_identity< SimdKokkosStream<float>::simd_t > {
  KOKKOS_FORCEINLINE_FUNCTION static SimdKokkosStream<float>::simd_t sum() {
    return SimdKokkosStream<float>::simd_t(1.0f);
  }
};
template<>
struct reduction_identity< SimdKokkosStream<double>::simd_t > {
  KOKKOS_FORCEINLINE_FUNCTION static SimdKokkosStream<double>::simd_t sum() {
    return SimdKokkosStream<double>::simd_t(1.0);
  }
};
}

// ====================================================================
// ====================================================================
// custom reducer for simd type
template <class T, class Space>
struct SimdReducer {
 public:

  using simd_t = typename SimdKokkosStream<T>::simd_t;
  using simd_storage_t = typename SimdKokkosStream<T>::simd_storage_t;

  // Required
  using reducer = SimdReducer<T, Space>;
  using value_type = simd_t;
  using value_type_storage = simd_storage_t;
  using result_view_type = Kokkos::View<value_type, Space, Kokkos::MemoryUnmanaged>;

 private:
  result_view_type value;

 public:
  KOKKOS_INLINE_FUNCTION
  SimdReducer(value_type& value_) : value(&value_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest = dest + src;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const {
    val = simd_t(0.0);
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return true; }
};

template<class T, class simd_type>
KOKKOS_INLINE_FUNCTION
simd_type
load(const T* ptr)
{
  return simd_type( ptr, simd::element_aligned_tag{} );
}

template<class T, class simd_type>
KOKKOS_INLINE_FUNCTION
void
store( T* ptr, const simd_type& simd )
{
    simd.copy_to( ptr, simd::element_aligned_tag{} );
}

template <class T>
SimdKokkosStream<T>::SimdKokkosStream(
        const unsigned int ARRAY_SIZE, const int device_index)
    : array_size_scalar(ARRAY_SIZE),
      array_size_vector(ARRAY_SIZE/simd_t::size()),
      d_a("d_a", array_size_scalar),
      d_b("d_b", array_size_scalar),
      d_c("d_c", array_size_scalar),
      hm_a(create_mirror_view(d_a)),
      hm_b(create_mirror_view(d_b)),
      hm_c(create_mirror_view(d_c))
{
}

template <class T>
SimdKokkosStream<T>::~SimdKokkosStream()
{
}

template <class T>
void SimdKokkosStream<T>::init_arrays(T initA, T initB, T initC)
{
  Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
  {
    const auto sindex = index*simd_t::size();
    store( &d_a(sindex) , simd_t(initA));
    store( &d_b(sindex) , simd_t(initB));
    store( &d_c(sindex) , simd_t(initC));
  });
  Kokkos::fence();
}

template <class T>
void SimdKokkosStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  deep_copy(hm_a, d_a);
  deep_copy(hm_b, d_b);
  deep_copy(hm_c, d_c);

  for(int ii = 0; ii < array_size_scalar; ++ii)
  {
    a[ii] = (hm_a)(ii);
    b[ii] = (hm_b)(ii);
    c[ii] = (hm_c)(ii);
  }
}

template <class T>
void SimdKokkosStream<T>::copy()
{

  Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
  {
    const auto sindex = index*simd_t::size();
    const auto value = load<T,simd_t>(&d_a(sindex));
    store( &d_c(sindex), value);
  });
  Kokkos::fence();
}

template <class T>
void SimdKokkosStream<T>::mul()
{

  const T scalar = startScalar;
  Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
  {
    const auto sindex = index*simd_t::size();
    const auto value = scalar*load<T,simd_t>(&d_c(sindex));
    store( &d_b(sindex), value);
  });
  Kokkos::fence();
}

template <class T>
void SimdKokkosStream<T>::add()
{
  Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
  {
    const auto sindex = index*simd_t::size();
    const auto value = load<T,simd_t>(&d_a(sindex)) + load<T,simd_t>(&d_b(sindex));
    store( &d_c(sindex), value);
  });
  Kokkos::fence();
}

template <class T>
void SimdKokkosStream<T>::triad()
{
  const T scalar = startScalar;
  Kokkos::parallel_for(array_size_vector, KOKKOS_CLASS_LAMBDA (const long index)
  {
    const auto sindex = index*simd_t::size();
    const auto value = load<T,simd_t>(&d_b(sindex)) + scalar*load<T,simd_t>(&d_c(sindex));
    store( &d_a(sindex), value);
  });
  Kokkos::fence();
}

template <class T>
T SimdKokkosStream<T>::dot()
{

  // This is still buggy when simd_t is simd_abi::native, but ok whi simd_abi::pack

  // simd_t sum = simd_t(0.0);

  // using SimdReducerResult = SimdReducer<T, Kokkos::DefaultExecutionSpace>;

  // Kokkos::parallel_reduce(array_size_vector, KOKKOS_LAMBDA (const long index, simd_t &tmp)
  // {
  //   const auto sindex = index*simd_t::size();
  //   const auto val = load<T,simd_t>(&d_a(sindex)) * load<T,simd_t>(&d_b(sindex));
  //   tmp = tmp + val;
  // }, SimdReducerResult(sum));

  // auto res_view = Kokkos::View<T*, Kokkos::HostSpace>( "res", simd_t::size() );
  // store(res_view.data(), sum);

  // // final horizontal reduction (should be done with simd operator)
  // T res=0;
  // for (int i = 0; i<simd_t::size(); ++i)
  //   res+=res_view(i);
  // return res;

  return 0;
}
