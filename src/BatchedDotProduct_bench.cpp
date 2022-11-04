#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>
// Include Kokkos Headers
#include<Kokkos_Core.hpp>

#include <cblas.h>


#if defined(KOKKOS_ENABLE_CUDA)
#include "CudaTimer.h"
#endif

#if defined( KOKKOS_ENABLE_HIP)
#include "HipTimer.h"
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
#include "OpenMPTimer.h"
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
#include "SimpleTimer.h"
#endif

// define default execution space timer
#if defined(KOKKOS_ENABLE_CUDA)
using Timer = CudaTimer;
#elif defined( KOKKOS_ENABLE_HIP)
using Timer = HipTimer;
#elif defined(KOKKOS_ENABLE_OPENMP)
using Timer = OpenMPTimer;
#else
using Timer = SimpleTimer;
#endif

/*
 * Use the same heuristic as in KokkosKernels to compute the requested number of teams per
 * dot.
 *
 * see
 * https://github.com/kokkos/kokkos-kernels/blob/master/src/blas/impl/KokkosBlas_util.hpp
 */
int computeNbTeamsPerDot(int vector_length, int nbDots)
{

  constexpr int workPerTeam = 4096;  // desired amount of work per team
  int teamsPerDot = 1;

  // approximate number of teams
  int approxNumTeams =
      (vector_length * nbDots) / workPerTeam;

  // Adjust approxNumTeams in case it is too small or too large
  if (approxNumTeams < 1)    approxNumTeams = 1;
  if (approxNumTeams > 1024) approxNumTeams = 1024;

  // If there are more reductions than the number of teams,
  // then set the number of teams to be number of reductions.
  // We don't want a team to contribute to more than one reduction.
  if (nbDots >= approxNumTeams) {
    teamsPerDot = 1;
  }
  // If there are more teams than reductions, each reduction can
  // potentially be performed by multiple teams. First, compute
  // teamsPerDot as an integer (take the floor, not ceiling), then,
  // compute actual number of teams by using this factor.
  else {
    teamsPerDot = approxNumTeams / nbDots;
  }

  return teamsPerDot;
}

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * Computed multiple dot product between column vectors taken from
 * input matrix X and Y. The result is a 1d vector res.
 * Parallel strategy is to use the team policy to perform the dot
 * products (reduction).
 *
 * \paran[in] X, Y : 2d input matrices
 * \param[out] res : 1d results
 *
 */
void batched_dot_product(int nx, int ny, int nrepeat, bool use_lambda)
{

  // Allocate Views
  Kokkos::View<double**, Kokkos::LayoutLeft> x("X", nx, ny);
  Kokkos::View<double**, Kokkos::LayoutLeft> y("Y", nx, ny);

  // vector of dot product, one per column, there are ny columns
  Kokkos::View<double*>  dotProd("dot_prod", ny);

  // Initialize arrays
  // first dot product should be near pi**2/6 ~ 1.64493 because
  // \Sum_{k=0}^{\infty} 1/k^2 = \frac{\pi^2}{6}
  {
    using md_policy = Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
    Kokkos::parallel_for(
      "init_arrays",
      md_policy({0,0},{nx,ny}),
      KOKKOS_LAMBDA (const int& i, const int& j)
      {
        x(i,j) = 1.0/(i+1)*(j+1);
        y(i,j) = 1.0/(i+1)*(j+1);
      });
  }

  // get prepared for TeamPolicy
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using member_t = team_policy_t::member_type;

  // number of teams is the number of dot product to perform
  int nbTeams = ny;

  // create a team policy for lambda
  const team_policy_t policy_lambda(nbTeams, Kokkos::AUTO(), Kokkos::AUTO());

  // define compute lambda
  auto dot_prod_lambda = KOKKOS_LAMBDA (const member_t& member)
    {
      // inside team, compute dot product as a parallel reduce operation
      double dot_prod = 0;
      int j = member.league_rank();

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, nx),
        [&](const int &i, double &update)
        {
          update += x(i,j) * y(i,j);
        },
        dot_prod);
      // only one thread per team, collect the final reduction result, and write it
      // the output view
     Kokkos::single(Kokkos::PerTeam(member), [&]() { dotProd(j) = dot_prod; });
    };

  int nbTeamsPerDot = computeNbTeamsPerDot(nx,ny);
  printf("Using nbTeamsPerDot = %d\n",nbTeamsPerDot);

  // number of teams is the number of dot product to perform
  int nbTeams2 = nbTeamsPerDot * ny;
  printf("Using nbTeams = %d\n", nbTeams2);

  // create a team policy for lambda
  const team_policy_t policy_lambda2(nbTeams2, Kokkos::AUTO());

  // define compute lambda for n teams per dot
  auto dot_prod_lambda2 = KOKKOS_LAMBDA (const member_t& member)
    {

      // inside a team, compute dot product partial result as a parallel reduce operation
      double partial_dot_prod = 0;
      int teamId = member.league_rank();

      // get column number
      int j         = teamId / nbTeamsPerDot;

      // get the pieceId inside a given column
      int pieceId   = teamId % nbTeamsPerDot;
      int pieceSize = nx / nbTeamsPerDot;

      // get the line index begin/end
      int begin     =  pieceId      * pieceSize;
      int end       = (pieceId + 1) * pieceSize;

      // the last piece might be slightly larger
      if (pieceId == nbTeamsPerDot - 1) end = nx;

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, begin, end),
        [&](const int &i, double &update)
        {
          update += x(i,j) * y(i,j);
        },
        partial_dot_prod);
      // only one thread per team, collect the final reduction result, and write it
      // the output view
     Kokkos::single(Kokkos::PerTeam(member),
                    [&]() { Kokkos::atomic_add(&dotProd(j), partial_dot_prod); });
    };


  /*
   * Kokkos one team per dot
   */
  {
    // Measure computation time
    Timer timer;

    timer.start();

    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, one team per dot-product
      Kokkos::parallel_for(
        "compute_dot_products_lambda",
        policy_lambda,
        dot_prod_lambda);
    }

    timer.stop();

    double time_seconds = timer.elapsed();

    printf("Kokkos one team per dot:\n");
    printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
    printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
           nx, ny,
           time_seconds,
           time_seconds/nrepeat,
           (nx*ny*2+ny)*sizeof(double)*1.0e-6,
           (nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);
    // print results
    // {
    //   auto dotProd_h = Kokkos::create_mirror_view(dotProd);
    //   Kokkos::deep_copy(dotProd_h, dotProd);
    //   for (int j=0; j<4; ++j)
    //     printf("dotProd(j=%d)=%f\n",j,dotProd_h(j));
    // }
  }

  /*
   * Kokkos n teams per dot
   */
  {
    Kokkos::deep_copy(dotProd, 0.0);

    // Measure computation time
    Timer timer;

    timer.start();

    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, one team per dot-product
      Kokkos::parallel_for(
        "compute_dot_products_lambda2",
        policy_lambda2,
        dot_prod_lambda2);
    }

    timer.stop();

    double time_seconds = timer.elapsed();

    printf("Kokkos %d team per dot:\n",nbTeamsPerDot);
    printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
    printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
           nx, ny,
           time_seconds,
           time_seconds/nrepeat,
           (nx*ny*2+ny)*sizeof(double)*1.0e-6,
           (nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);
    // print results
    // {
    //   auto dotProd_h = Kokkos::create_mirror_view(dotProd);
    //   Kokkos::deep_copy(dotProd_h, dotProd);
    //   for (int j=0; j<4; ++j)
    //     printf("dotProd(j=%d)=%f\n",j,dotProd_h(j)/nrepeat);
    // }
  }

  /*
   * Using cblas on cpu with open parallelization for the loop on columns
   */
#ifdef USE_CBLAS_SERIAL
  {

    // Allocate Views in OpenMP execspace
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::OpenMP> x2("X2", nx, ny);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::OpenMP> y2("Y2", nx, ny);

    // vector of dot product, one per column, there are ny columns
    Kokkos::View<double*, Kokkos::OpenMP>                      dotProd2("dot_prod2", ny);

    // Initialize arrays
    // first dot product should be near pi**2/6 ~ 1.64493 because
    // \Sum_{k=0}^{\infty} 1/k^2 = \frac{\pi^2}{6}
    {
      using md_policy = Kokkos::MDRangePolicy< Kokkos::OpenMP, Kokkos::Rank<2> >;
      Kokkos::parallel_for(
        "init_arrays",
        md_policy({0,0},{nx,ny}),
        KOKKOS_LAMBDA (const int& i, const int& j)
        {
          x2(i,j) = 1.0/(i+1)*(j+1);
          y2(i,j) = 1.0/(i+1)*(j+1);
      });
    }


    OpenMPTimer timer;

    timer.start();

    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, one team per dot-product
#pragma omp parallel for
      for (int j=0; j<ny; ++j)
      {
        const double *px = &x2(0,j);
        const double *py = &y2(0,j);
        double * pdot = dotProd2.data();

        dotProd2[j] = cblas_ddot(nx, px, 1, py, 1);
      }

    } // end for k

    timer.stop();
    double time_seconds = timer.elapsed();

    printf("CBLAS serial:\n");
    printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
    printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
           nx, ny,
           time_seconds,
           time_seconds/nrepeat,
           (nx*ny*2+ny)*sizeof(double)*1.0e-6,
           (nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);

    // print results
    // {
    //   for (int j=0; j<4; ++j)
    //     printf("dotProd(j=%d)=%f\n",j,dotProd2(j));
    // }

  }
#endif // USE_CBLAS_SERIAL

} // batched_dot_product

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char* argv[]) {

  // Parameters
  int nx = 1000;          // length of column vectors
  int ny = 100;           // number of column vectors
  int nrepeat = 10;       // number of invocations
  bool use_lambda = true; // use  lambda or functor

  // Read command line arguments
  for(int i=0; i<argc; i++) {
    if( strcmp(argv[i], "-nx") == 0) {
      nx = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-ny") == 0) {
      ny = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-use_lambda") == 0) {
      int tmp = atoi(argv[++i]);
      use_lambda = tmp!=0 ? true : false;
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("Batched Dot Products options:\n");
      printf("  -nx <int>:         length of column vectors (default: 1000)\n");
      printf("  -ny <int>:         number of column vectors (default: 10)\n");
      printf("  -nrepeat <int>:    number of integration invocations (default: 10)\n");
      printf("  -use_lambda <int>: use lambda ? (default: 1)\n");
      printf("  -help (-h):        print this message\n");
    }
  }

  if (use_lambda)
    printf("Using lambda  version\n");
  else
    printf("Using functor version\n");

  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  // run test
  batched_dot_product(nx, ny, nrepeat, use_lambda);

  // Shutdown Kokkos
  Kokkos::finalize();
}
