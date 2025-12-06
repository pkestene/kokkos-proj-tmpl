#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <cstdint>

// Include Kokkos Headers
#include <Kokkos_Core.hpp>

#include <Kokkos_Random.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#  include "CudaTimer.h"
using Timer = CudaTimer;
#elif defined(KOKKOS_ENABLE_HIP)
#  include "HipTimer.h"
using Timer = HipTimer;
#elif defined(KOKKOS_ENABLE_OPENMP)
#  include "HostTimer.h"
using Timer = HostTimer;
#else
#  include "HostTimer.h"
using Timer = HostTimer;
#endif

// ===============================================================
// ===============================================================
// ===============================================================
void
test_view()
{

  using View_t = Kokkos::View<int **, Kokkos::LayoutLeft>;

  View_t view("view", 5, 10);

  printf("is View_t layout left ? %d\n",
         std::is_same<typename View_t::traits::array_layout, Kokkos::LayoutLeft>::value);

  View_t view2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "view2"), 5, 10);

  constexpr int layout2 = view2.layout();

  // printf("is View_t layout left ? %d\n", std::is_same<view2.layout(),
  // Kokkos::LayoutLeft>::value);

} // test_view

// ===============================================================
// ===============================================================
// ===============================================================
int
main(int argc, char * argv[])
{

  // Parameters
  int niter = 10000; // number of iterations for the parallel_reduce loop
  int nrepeat = 10;  // number of random generator draw per thread

  // Read command line arguments
  for (int i = 0; i < argc; i++)
  {
    if (strcmp(argv[i], "-niter") == 0)
    {
      niter = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "-nrepeat") == 0)
    {
      nrepeat = atoi(argv[++i]);
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      printf("Compute pi Options:\n");
      printf("  -niter <int>:     number of iteration (default: 10000)\n");
      printf("  -nrepeat <int>:   number of rand gen draws per thread (default: 10)\n");
      printf("  -help (-h):       print this message; the total number of points generated is "
             "niter*nrepeat\n");
    }
  }

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  // run test
  test_view();

  // Shutdown Kokkos
  Kokkos::finalize();
}
