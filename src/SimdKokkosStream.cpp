// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "SimdKokkosStream.hpp"

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

template class SimdKokkosStream<float>;
template class SimdKokkosStream<double>;
