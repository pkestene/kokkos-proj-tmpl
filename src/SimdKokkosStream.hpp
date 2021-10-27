// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

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
