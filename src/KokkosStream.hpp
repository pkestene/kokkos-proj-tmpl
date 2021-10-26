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

#include "Stream.h"

//#define IMPLEMENTATION_STRING "Kokkos"

template <class T>
class KokkosStream : public Stream<T>
{
protected:
  // Size of arrays
  unsigned int array_size;

  using view_t = Kokkos::View<T*>;
  using mirror_view_t = typename view_t::HostMirror;

  // Device side pointers to arrays
  view_t d_a;
  view_t d_b;
  view_t d_c;
  mirror_view_t hm_a;
  mirror_view_t hm_b;
  mirror_view_t hm_c;

public:

  KokkosStream(const unsigned int, const int);
  ~KokkosStream();

  virtual void copy() override;
  virtual void add() override;
  virtual void mul() override;
  virtual void triad() override;
  virtual T dot() override;

  virtual void init_arrays(T initA, T initB, T initC) override;
  virtual void read_arrays(
    std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

