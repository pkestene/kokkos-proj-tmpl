// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "KokkosStream.hpp"


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
