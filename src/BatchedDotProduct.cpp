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

class DotProdFunctor
{
 public:

  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using member_t = team_policy_t::member_type;

  DotProdFunctor(
    Kokkos::View<double**, Kokkos::LayoutLeft> x,
    Kokkos::View<double**, Kokkos::LayoutLeft> y,
    Kokkos::View<double*, Kokkos::LayoutLeft> dotProd)
  : m_x(x),
    m_y(y),
    m_dotProd(dotProd),
    m_nx(x.extent(0)),
    m_ny(x.extent(1))
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_t& member) const
  {
    // inside team, compute dot product as a parallel reduce operation
    double dot_prod = 0;
    int j = member.league_rank();

    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(member, m_nx),
      [&](const int &i, double &update)
      {
        update += m_x(i,j) * m_y(i,j);
      },
      dot_prod);
    // only one thread per team, collect the final reduction result, and write it
    // the output view
    Kokkos::single(Kokkos::PerTeam(member), [&]() { m_dotProd(j) = dot_prod; });

  }

  inline void run()
  {

    // team size (number of threads per team)
    const int team_size_max = DotProdFunctor::team_policy_t(m_ny, 1).team_size_max(
      *this, Kokkos::ParallelForTag());

    //const team_policy_t policy(m_ny, team_size_max);
    const team_policy_t policy(m_ny, Kokkos::AUTO(), Kokkos::AUTO());

    Kokkos::parallel_for(
      "compute_dot_products_functor",
      policy,
      *this);

  }

  Kokkos::View<double**, Kokkos::LayoutLeft> m_x;
  Kokkos::View<double**, Kokkos::LayoutLeft> m_y;
  Kokkos::View<double*, Kokkos::LayoutLeft>  m_dotProd;
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

  // create functor to compute batched dot product
  DotProdFunctor dot_prod_functor(x,y,dotProd);

  // team size (number of threads per team) for functor version
  const int team_size_max = team_policy_t(ny, 1).team_size_max(
    dot_prod_functor, Kokkos::ParallelForTag());

  // create a team policy for functor
  //const team_policy_t policy_functor(ny, team_size_max);
  const team_policy_t policy_functor(ny, Kokkos::AUTO(), Kokkos::AUTO());

  // Measure computation time
  Timer timer;

  timer.start();

  if (use_lambda)
  {
    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, one team per dot-product
      Kokkos::parallel_for(
        "compute_dot_products_lambda",
        policy_lambda,
        dot_prod_lambda);
    }
  }
  else // use functor version
  {
    for(int k = 0; k < nrepeat; k++)
    {
      // Do batched dot product
      // using hierarchical parallelism, one team per dot-product
      Kokkos::parallel_for(
        "compute_dot_products_functor",
        policy_functor,
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
