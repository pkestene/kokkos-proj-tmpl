#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>
// Include Kokkos Headers
#include<Kokkos_Core.hpp>


#ifdef KOKKOS_ENABLE_CUDA
#include "CudaTimer.h"
using Timer = CudaTimer;
#elif defined( KOKKOS_ENABLE_HIP)
#include "HipTimer.h"
using Timer = HipTimer;
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "OpenMPTimer.h"
using Timer = OpenMPTimer;
#else
#include "SimpleTimer.h"
using Timer = SimpleTimer;
#endif

using DataType = uint64_t;
using ViewTypeSrc = Kokkos::View<DataType*,Kokkos::DefaultExecutionSpace>;
using ViewTypeDst = Kokkos::View<DataType*,Kokkos::DefaultExecutionSpace>;

// ===============================================================
// ===============================================================
// ===============================================================
void test_deep_copy_functor(int length, int nrepeat) {

  // Allocate Views
  ViewTypeSrc x("X",length);
  ViewTypeDst y("Y",length);
  
  // Initialize arrays
  Kokkos::parallel_for(length, KOKKOS_LAMBDA (const size_t& i) {
    x(i) = 1;
    y(i) = 1;
  });

  // Time computation
  Timer timer;

  timer.start();
  for(int k = 0; k < nrepeat; k++) {

    // Do compute
    Kokkos::parallel_for(length, KOKKOS_LAMBDA (const size_t& i) {
      y(i) = x(i);
    });

  }
  timer.stop();
  
  // Print results
  double time_seconds = timer.elapsed();

  printf("# DEEP COPY FUNCTOR ###############################################\n");
  printf("# VectorLength  Time(s) TimePerIterations(s)    size(MB)   BW(GB/s)\n");
  printf(" %13i %8lf %20.3e  %10.2f %10.2f\n",length,time_seconds,time_seconds/nrepeat,1.0e-6*length*2*sizeof(DataType),1.0e-9*length*2*nrepeat*sizeof(DataType)/time_seconds);

} // test_deep_copy_functor

// ===============================================================
// ===============================================================
// ===============================================================
void test_deep_copy_api(int length, int nrepeat) {

  // Allocate Views
  ViewTypeSrc x("X",length);
  ViewTypeDst y("Y",length);
  
  // Initialize arrays
  Kokkos::parallel_for(length, KOKKOS_LAMBDA (const size_t& i) {
    x(i) = 1;
    y(i) = 1;
  });

  // Time computation
  Timer timer;

  timer.start();
  for(int k = 0; k < nrepeat; k++) {

    // Do compute
    Kokkos::deep_copy(y,x);
    //Kokkos::Impl::view_copy(y,x);
    
  }
  //Kokkos::fence();
  timer.stop();
  
  // Print results
  double time_seconds = timer.elapsed();

  printf("# DEEP COPY API     ###############################################\n");
  printf("# VectorLength  Time(s) TimePerIterations(s)    size(MB)   BW(GB/s)\n");
  printf(" %13i %8lf %20.3e  %10.2f %10.2f\n",length,time_seconds,time_seconds/nrepeat,1.0e-6*length*2*sizeof(DataType),1.0e-9*length*2*nrepeat*sizeof(DataType)/time_seconds);

} // test_deep_copy_api

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char* argv[]) {

  // Parameters
  int length = 1<<26; // length of vectors
  int nrepeat = 10;     // number of integration invocations

  // Read command line arguments
  for(int i=0; i<argc; i++) {
    if( strcmp(argv[i], "-l") == 0) {
      length = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("Deep copy Options:\n");
      printf("  -l <int>:         length of vectors (default: 10000000)\n");
      printf("  -nrepeat <int>:   number of integration invocations (default: 10)\n");
      printf("  -help (-h):       print this message\n");
    }
  }
  
  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  // run test
  test_deep_copy_functor(length, nrepeat);
  test_deep_copy_api    (length, nrepeat);

  // Shutdown Kokkos
  Kokkos::finalize();
}
