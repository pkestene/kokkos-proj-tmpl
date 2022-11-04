/*
 * Same as BatchedDotProduct.cpp but running multiple teams per dot product.
 */

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

class DotProdFunctor
{
 public:

  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using member_t = team_policy_t::member_type;

  DotProdFunctor(
    Kokkos::View<double**, Kokkos::LayoutLeft> x,
    Kokkos::View<double**, Kokkos::LayoutLeft> y,
    Kokkos::View<double*, Kokkos::LayoutLeft> dotProd,
    int teamsPerDot)
  : m_x(x),
    m_y(y),
    m_dotProd(dotProd),
    m_teamsPerDot(teamsPerDot),
    m_nx(x.extent(0)),
    m_ny(x.extent(1))
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_t& member) const
  {
    // inside a team, compute dot product partial result as a parallel reduce operation
    double partial_dot_prod = 0;
    int teamId = member.league_rank();

    // get column number
    int j        = teamId / m_teamsPerDot;

    // get the pieceId inside a given column
    int pieceId   = teamId % m_teamsPerDot;
    int pieceSize = m_x.extent(0) / m_teamsPerDot;

    // get the line index begin/end
    int begin     =  pieceId      * pieceSize;
    int end       = (pieceId + 1) * pieceSize;

    // the last piece might be slightly larger
    if (pieceId == m_teamsPerDot - 1) end = m_x.extent(0);

    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(member, begin, end),
      [&](const int &i, double &update)
      {
        update += m_x(i,j) * m_y(i,j);
      },
      partial_dot_prod);
    // only one thread per team will add (atomically) its partial result to the global results
    // in the output view
    // since multiple teams contribute to a given dot product, the partial results MUST be
    // added using an atomic addition
    Kokkos::single(Kokkos::PerTeam(member),
                   [&]() { Kokkos::atomic_add(&m_dotProd(j), partial_dot_prod); });

  } // operator()

  inline void run()
  {

    int nbTeams = m_teamsPerDot * m_ny;

    // create a team policy for our functor
    const team_policy_t policy(nbTeams, Kokkos::AUTO());

    Kokkos::parallel_for(
      "compute_dot_products_functor",
      policy,
      *this);

  } // run

  Kokkos::View<double**, Kokkos::LayoutLeft> m_x;
  Kokkos::View<double**, Kokkos::LayoutLeft> m_y;
  Kokkos::View<double*, Kokkos::LayoutLeft>  m_dotProd;
  int m_teamsPerDot;
  int m_nx, m_ny;

}; // class DotProdFunctor


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
  Kokkos::View<double*, Kokkos::LayoutLeft>  dotProd("dot_prod", ny);

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

  // zeroing dotProd
  Kokkos::deep_copy(dotProd, 0.0);

  // get prepared for TeamPolicy
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using member_t = team_policy_t::member_type;

  int nbTeamsPerDot = computeNbTeamsPerDot(nx,ny);
  printf("Using nbTeamsPerDot = %d\n",nbTeamsPerDot);

  int nbTeams = nbTeamsPerDot * ny;
  printf("Using nbTeams = %d\n", nbTeams);

  // create a team policy
  const team_policy_t policy(nbTeams, Kokkos::AUTO());

  // define compute lambda
  auto dot_prod_lambda = KOKKOS_LAMBDA (const member_t& member)
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

  // create functor to compute batched dot product
  DotProdFunctor dot_prod_functor(x,y,dotProd,nbTeamsPerDot);

  // Measure computation time
  Timer timer;

  timer.start();

  if (use_lambda)
  {
    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, multiple teams per dot-product
      Kokkos::parallel_for(
        "compute_dot_products_lambda",
        policy,
        dot_prod_lambda);
    }
  }
  else // use functor version
  {
    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, multiple teams per dot-product
      Kokkos::parallel_for(
        "compute_dot_products_functor",
        policy,
        dot_prod_functor);

      //dot_prod_functor.run();
    }
  }

  timer.stop();

  // Print results
  double time_seconds = timer.elapsed();

  // print results
  // {
  //   auto dotProd_h = Kokkos::create_mirror_view(dotProd);
  //   Kokkos::deep_copy(dotProd_h, dotProd);
  //   for (int j=0; j<ny; ++j)
  //     printf("dotProd(j=%d)=%f\n",j,dotProd_h(j));
  // }

  printf("#nx      ny        Time(s) TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%7i %7i   %8lf %20.3e  %3.3f %3.3f\n",
         nx, ny,
         time_seconds,
         time_seconds/nrepeat,
         (nx*ny*2+ny)*sizeof(double)*1.0e-6,
         (nx*ny*2+ny)*sizeof(double)*nrepeat/time_seconds*1.0e-9);

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
