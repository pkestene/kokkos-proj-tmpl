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
#  include "OpenMPTimer.h"
using Timer = OpenMPTimer;
#else
#  include "SimpleTimer.h"
using Timer = SimpleTimer;
#endif

/*
 * Here is a simple example to illutrate how to use Kokkos random
 * generator / pool.
 *
 * Compute pi estimate :
 * generate a large numpber of points p = (x,y) in the square [0,1]^2
 * and count (reduce) the number of points that falls inside the circle
 * of radius 1. The ratio will provide an estimes of pi/4.
 *
 * We use a two level hierarchy to generate the points
 * parameter:
 * int niter   : number of iterations
 * int nrepeat : number of points generated by a given thread
 *
 * The total number of points generated will be niter*nrepeat.
 */

// ===============================================================
/**
 * Functor to estimate pi.
 *
 * \tparam GeneratorPool define the type of random generator (e.g.
 *         Kokkos::Random_XorShift64_Pool, Kokkos::Random_XorShift1024_Pool, ..)
 * Note that we explicitely use scalar type = double as the type of scalar
 * drawn by the random number generator.
 *
 */
template <class GeneratorPool>
struct compute_pi_functor
{

  // define some type alias

  // type for hold the random generator state
  using rnd_t = typename GeneratorPool::generator_type;

  // which execution space ? OpenMP, Cuda, ...
  using device_t = typename GeneratorPool::device_type;

  // a random generator pool object
  GeneratorPool rand_pool;

  int niter;
  int nrepeat;

  /**
   * constructor.
   *
   * The randomm generator pool must have been initialized in the calling
   * routine
   */
  compute_pi_functor(GeneratorPool rand_pool_, int niter_, int nrepeat_)
    : rand_pool(rand_pool_)
    , niter(niter_)
    , nrepeat(nrepeat_)
  {}

  static double
  apply(GeneratorPool rand_pool_, int niter_, int nrepeat_)
  {

    // create functor
    compute_pi_functor functor(rand_pool_, niter_, nrepeat_);

    // initialize the reduce variable
    uint64_t nb_pt_inside = 0;

    // perform computation
    Kokkos::parallel_reduce(niter_, functor, nb_pt_inside);

    // final results
    double pi = 4.0 * nb_pt_inside / (niter_ * nrepeat_);

    return pi;
  }

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int i, uint64_t & count) const
  {

    uint32_t local_count = 0;

    rnd_t rand_gen = rand_pool.get_state();
    for (int k = 0; k < nrepeat; ++k)
    {

      // draw a point (in square [0,1]^2)
      const double x = Kokkos::rand<rnd_t, double>::draw(rand_gen);
      const double y = Kokkos::rand<rnd_t, double>::draw(rand_gen);

      double r2 = x * x + y * y;
      if (r2 < 1.0)
        local_count++;
    }

    count += local_count;

    // free random gen state, so that it can used by other threads later.
    rand_pool.free_state(rand_gen);

  } // operator ()

}; // compute_pi_functor

// ===============================================================
// ===============================================================
// ===============================================================
void
compute_pi(int niter, int nrepeat)
{

  // define an alias to the random generator pool
  using RGPool_t = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

  // initialize the random generator pool
  uint64_t ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "Test Seed:" << ticks << std::endl;

  RGPool_t pool(ticks);

  // Time computation
  Timer timer;

  timer.start();
  double pi = compute_pi_functor<RGPool_t>::apply(pool, niter, nrepeat);
  timer.stop();

  // Print results
  double time_seconds = timer.elapsed();

  printf("pi(estimate) = %f\n", pi);
  printf("(pi(estimated)-pi)/pi = %8.7g\n", fabs(pi - M_PI) / M_PI);
  printf("time in seconds: %g\n", time_seconds);

} // compute_pi

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
  compute_pi(niter, nrepeat);

  // Shutdown Kokkos
  Kokkos::finalize();
}
