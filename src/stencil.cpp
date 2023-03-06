/**
 * stencil kernel: 7 points
 *
 * Please note we measure performance using an effective metric:
 * number of read/write operations which involve a memory transaction
 * without (as much as possible) re-using cached data.
 *
 * Here we chose to count all neighbors memory access, deliberately knowing
 * that we over-estimate the actual bandwidth.
 */

#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>
#include <vector>
#include <fstream>      // std::ofstream

// Include Kokkos Headers
#include<Kokkos_Core.hpp>

// make the compiler ignore an unused variable
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

#ifdef USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif // USE_DOUBLE

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

using Device = Kokkos::DefaultExecutionSpace;
using DataArray = Kokkos::View<real_t***, Device>;
using DataArray1d = Kokkos::View<real_t*, Device>;
using DataArrayR = Kokkos::View<real_t***, Kokkos::LayoutRight, Device>;

KOKKOS_INLINE_FUNCTION
double init_x(int i, int j, int k) {
  return 1.0*(i+j*0.12345+k*0.31415+0.1);
}

KOKKOS_INLINE_FUNCTION
double init_y(int i, int j, int k) {
  return 3.0*(i+j+k);
}

// ===============================================================
// ===============================================================
KOKKOS_INLINE_FUNCTION
void index2coord(int index,
                 int &i, int &j,
                 int Nx, int Ny)
{
  UNUSED(Nx);
#ifdef KOKKOS_ENABLE_CUDA
  j = index / Nx;
  i = index - j*Nx;
#else
  i = index / Ny;
  j = index - i*Ny;
#endif
} // index2coord - 2d

// ===============================================================
// ===============================================================
KOKKOS_INLINE_FUNCTION
void index2coord(int index,
                 int &i, int &j, int &k,
                 int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  int NxNy = Nx*Ny;
  k = index / NxNy;
  j = (index - k*NxNy) / Nx;
  i = index - j*Nx - k*NxNy;
#else
  int NyNz = Ny*Nz;
  i = index / NyNz;
  j = (index - i*NyNz) / Nz;
  k = index - j*Nz - i*NyNz;
#endif
} // index2coord - 3d

// ===============================================================
// ===============================================================
KOKKOS_INLINE_FUNCTION
int INDEX(int i,  int j,  int k,
          int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  return i + Nx*j + Nx*Ny*k; // left layout
#else
  return k + Nz*j + Nz*Ny*i; // right layout
#endif
} // INDEX

// ===============================================================
// ===============================================================
KOKKOS_INLINE_FUNCTION
int INDEX(int i,  int j,
          int Nx, int Ny)
{
  UNUSED(Nx);
#ifdef KOKKOS_ENABLE_CUDA
  return i + Nx*j; // left layout
#else
  return j + Ny*i; // right layout
#endif
} // INDEX

// ===============================================================
// ===============================================================
/**
 * Right Layout linearization
 */
KOKKOS_INLINE_FUNCTION
int RINDEX(int i,  int j,  int k,
           int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
  return k + Nz*j + Nz*Ny*i; // right layout
} // RINDEX

// ===============================================================
// ===============================================================
KOKKOS_INLINE_FUNCTION
void index2coord_right(int index,
                       int &i, int &j, int &k,
                       int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
  int NyNz = Ny*Nz;
  i = index / NyNz;
  j = (index - i*NyNz) / Nz;
  k = index - j*Nz - i*NyNz;
} // index2coord_right - 3d

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 1 : naive
 * - data is a 3d array
 * - all loops parallelized with a single parallel_for
 *
 * \return effective bandwidth
 */
double test_stencil_3d_flat(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;

  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // Initialize arrays
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA (const int& index) {
      int i,j,k;

      index2coord(index,i,j,k,n,n,n);

      x(i,j,k) = init_x(i,j,k);
      y(i,j,k) = init_y(i,j,k);
    });

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA (const int& index) {
	int i,j,k;
	index2coord(index,i,j,k,n,n,n);

	if (i>0 and i<n-1 and
	    j>0 and j<n-1 and
	    k>0 and k<n-1 )
	y(i,j,k) = -5*x(i,j,k) +
	  ( x(i-1,j,k) + x(i+1,j,k) +
	    x(i,j-1,k) + x(i,j+1,k) +
	    x(i,j,k-1) + x(i,j,k+1) );
      });

  }
  timer.stop();

  // for (int i=13; i<17; ++i) {
  //   for (int j=13; j<17; ++j) {
  //     for (int k=13; k<17; ++k) {
  //       printf("%f ",y(i,j,k));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

    // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_flat

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 1bis : naive
 * - data is a 1d array, mapping between (i,j,k) and memory address is
 *   handled at application level.
 * - all loops parallelized with a single parallel_for with 1d range
 *   policy
 *
 * \return effective bandwidth
 */
double test_stencil_3d_flat_1d_array(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;
  uint64_t nbCellsXY = n*n;

  // Allocate Views
  DataArray1d x("X",nbCells);
  DataArray1d y("Y",nbCells);

  // Initialize arrays
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA (const int& index) {
      int i,j,k;

      index2coord(index,i,j,k,n,n,n);

      x(index) = init_x(i,j,k);
      y(index) = init_y(i,j,k);
    });

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for(nbCellsXY, KOKKOS_LAMBDA (const int& index) {
	int i,j;
	index2coord(index,i,j,n,n);

	// int dx = n*n;
	// int dy = n;
	// int dz = 1;

	if (i>0 and i<n-1 and
	    j>0 and j<n-1 )

#if defined( __INTEL_COMPILER )
#pragma ivdep
#pragma omp simd
#endif
          for (int k=1; k<n-1; ++k) {

	    // int index = k + n*j + n*n*i;

            // y(index) = -5*x(index) +
            //   ( x(index-dx) + x(index+dx) +
            //     x(index-dy) + x(index+dy) +
            //     x(index-dz) + x(index+dz) );

            // y(k+n*j+n*n*i) = -5*x(k+n*j+n*n*i) +
            //   ( x(k+n*j+n*n*(i-1)) + x(k+n*j+n*n*(i+1)) +
            //     x(k+n*(j-1)+n*n*i) + x(k+n*(j+1)+n*n*i) +
            //     x(k+1+n*j+n*n*i)   + x(k-1+n*j+n*n*i) );

            y(INDEX(i,j,k,n,n,n)) = -5*x(INDEX(i,j,k,n,n,n)) +
              ( x(INDEX(i-1,j,k,n,n,n)) + x(INDEX(i+1,j,k,n,n,n)) +
                x(INDEX(i,j-1,k,n,n,n)) + x(INDEX(i,j+1,k,n,n,n)) +
                x(INDEX(i,j,k-1,n,n,n)) + x(INDEX(i,j,k+1,n,n,n)) );
          }
      });

  }
  timer.stop();

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_flat_1d_array

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 2:
 * - data is a 3d array
 * - only loops over i,j are parallelized, loop over k is kept inside
 * - optionally, on can use 1D subview to access data, and help the
 *   compiler to recognize a vectorizable loop
 *
 *
 * subview's are important here.
 * Without 1d views, you'll have about 25% percent perfomance drop here.
 *
 * Use parameter use_1d_views to activate/deactivate the use of 1d views.
 */
double test_stencil_3d_flat_vector(int n, int nrepeat, bool use_1d_views) {

  uint64_t nbCells = n*n*n;

  uint64_t nbIter = n*n;

  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // Initialize arrays
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA (const int& index) {
      int i,j,k;

      index2coord(index,i,j,k,n,n,n);

      x(i,j,k) = init_x(i,j,k);
      y(i,j,k) = init_y(i,j,k);
    });

  // Time computation
  Timer timer;

  timer.start();

  if (use_1d_views) {

    for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

      // Do stencil
      Kokkos::parallel_for(nbIter, KOKKOS_LAMBDA (const int& index) {
	  int i,j;

	  // index = j + n * i -- CPU
	  // index = i + n * j -- GPU
	  index2coord(index,i,j,n,n);

	  auto x_i_j = Kokkos::subview(x, i, j, Kokkos::ALL());
	  auto y_i_j = Kokkos::subview(y, i, j, Kokkos::ALL());

	  auto x_im1_j = Kokkos::subview(x, i-1, j, Kokkos::ALL());
	  auto x_ip1_j = Kokkos::subview(x, i+1, j, Kokkos::ALL());

	  auto x_i_jm1 = Kokkos::subview(x, i, j-1, Kokkos::ALL());
	  auto x_i_jp1 = Kokkos::subview(x, i, j+1, Kokkos::ALL());

	  /*
	   * subview's are important here. If you uncomment the following lines
	   * you'll about 25% percent perfomance drop here.
	   */
	  if (i>0 and i<n-1 and
	      j>0 and j<n-1) {

	    // vectorization loop
#if defined( __INTEL_COMPILER )
#pragma ivdep
#endif
	    for (int k=1; k<n-1; ++k) {
	      // y(i,j,k) = -5*x(i,j,k) +
	      //   ( x(i-1,j,k) + x(i+1,j,k) +
	      // 	x(i,j-1,k) + x(i,j+1,k) +
	      // 	x(i,j,k-1) + x(i,j,k+1) );
	      y_i_j(k) = -5*x_i_j(k) +
		( x_im1_j(k) + x_ip1_j(k) +
		  x_i_jm1(k) + x_i_jp1(k) +
		  x_i_j(k-1) + x_i_j(k+1) );
	    }
	  }

	});

    } // end repeat

  } else { // don't use 1 d views

    for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

      // Do stencil
      Kokkos::parallel_for(nbIter, KOKKOS_LAMBDA (const int& index) {
	  int i,j;
	  //index2coord(index,i,j,k,n,n,n);

	  // index = j + n * i -- CPU
	  // index = i + n * j -- GPU
	  index2coord(index,i,j,n,n);

	  if (i>0 and i<n-1 and
	      j>0 and j<n-1) {

	    // vectorization loop
#if defined( __INTEL_COMPILER )
#pragma ivdep
#endif
	    for (int k=1; k<n-1; ++k) {
	      y(i,j,k) = -5*x(i,j,k) +
	        ( x(i-1,j,k) + x(i+1,j,k) +
                  x(i,j-1,k) + x(i,j+1,k) +
                  x(i,j,k-1) + x(i,j,k+1) );
	    }
	  }

	});
    } // end repeat
  }
  timer.stop();

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_flat_vector

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 3 :
 * same as version 1 (naive) but uses a 3d range policy.
 */
double test_stencil_3d_range(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;

  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // init 3d range policy
#if KOKKOS_VERSION_MAJOR > 3
  using Range3D = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >;
#else
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;
#endif

  Range3D range( {{0,0,0}}, {{n,n,n}} );


  // Initialize arrays
  Kokkos::parallel_for("init", range,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {
                         x(i,j,k) = init_x(i,j,k);
                         y(i,j,k) = init_y(i,j,k);
		       });

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for
      ("stencil compute - 3d range", range, KOKKOS_LAMBDA (const int& i,
							   const int& j,
							   const int& k) {
	if (i>0 and i<n-1 and
	    j>0 and j<n-1 and
	    k>0 and k<n-1 )
	  y(i,j,k) = -5*x(i,j,k) +
	    ( x(i-1,j,k) + x(i+1,j,k) +
	      x(i,j-1,k) + x(i,j+1,k) +
	      x(i,j,k-1) + x(i,j,k+1) );
      });

  }
  timer.stop();

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_range

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 4 :
 * same as version 3 but uses a 2d Range policy and keep the loop over
 * index k inside kernel for CPU vectorization.
 */
double test_stencil_3d_range_vector(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;

  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // init 2d range policy
#if KOKKOS_VERSION_MAJOR > 3
  using Range2D = typename Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
  using Range3D = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >;
#else
  using Range2D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<2> >;
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;
#endif


  Range2D range2d( {{0,0}}, {{n,n}} );
  Range3D range3d( {{0,0,0}}, {{n,n,n}} );


  // Initialize arrays
  Kokkos::parallel_for("init", range3d,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {
                         x(i,j,k) = init_x(i,j,k);
                         y(i,j,k) = init_y(i,j,k);
		       });

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for
      ("stencil compute - 3d range vector", range2d, KOKKOS_LAMBDA (const int& i,
								    const int& j) {

	auto x_i_j = Kokkos::subview(x, i, j, Kokkos::ALL());
	auto y_i_j = Kokkos::subview(y, i, j, Kokkos::ALL());

	auto x_im1_j = Kokkos::subview(x, i-1, j, Kokkos::ALL());
	auto x_ip1_j = Kokkos::subview(x, i+1, j, Kokkos::ALL());

	auto x_i_jm1 = Kokkos::subview(x, i, j-1, Kokkos::ALL());
	auto x_i_jp1 = Kokkos::subview(x, i, j+1, Kokkos::ALL());

	if (i>0 and i<n-1 and
	    j>0 and j<n-1) {

	  // vectorization loop
#if defined( __INTEL_COMPILER )
#pragma ivdep
#endif
	  for (int k=1; k<n-1; ++k)
	    y_i_j(k) = -5*x_i_j(k) +
	      ( x_im1_j(k) + x_ip1_j(k) +
		x_i_jm1(k) + x_i_jp1(k) +
		x_i_j(k-1) + x_i_j(k+1) );
	}
      });

  }
  timer.stop();

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_range_vector

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 5 :
 * same as version 4 but uses Hierarchical parallelism, i.e.
 * - a TeamPolicy        Kokkos::policy for the outer loop
 * - a TeamThreadRange   Kokkos::policy to divide middle loop over threads
 * - a ThreadVectorRange Kokkos::policy for the inner loop (for CPU vectorization or GPU warp parallelism)
 *
 * Note that this is the only test with 3d views and Kokkos::LayoutRight
 * for all devices. We try to optimize by ensuring the inner loop will
 * use contiguous memory data.
 *
 */
double test_stencil_3d_range_hierarchical(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;

  // Allocate Views - enforce LayoutRight to make sure the fastest index
  // is used for inner loop iteration
  DataArrayR x("X",n,n,n);
  DataArrayR y("Y",n,n,n);

  // init 2d range policy
#if KOKKOS_VERSION_MAJOR > 3
  using Range2D = typename Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
  using Range3D = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >;
#else
  using Range2D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<2> >;
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;
#endif

  Range2D range2d( {{0,0}}, {{n,n}} );
  Range3D range3d( {{0,0,0}}, {{n,n,n}} );

  // Initialize arrays using a 3d range policy
  Kokkos::parallel_for("init", range3d,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {
                         x(i,j,k) = init_x(i,j,k);
			 y(i,j,k) = init_y(i,j,k);
		       });

  // get prepared for TeamPolicy
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using thread_t = team_policy_t::member_type;
  int nbTeams = n;

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for(
      team_policy_t(nbTeams,
                    Kokkos::AUTO, /* team size chosen by kokkos */
                    team_policy_t::vector_length_max()),
      KOKKOS_LAMBDA(const thread_t& thread) {

        int i = thread.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(thread, 1, n-1),
          [=](const int &j) {

            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(thread, 1, n-1),
              [=](const int &k) {

                if (i>0 and i<n-1 and
                    j>0 and j<n-1) {

                  // if (i==1 and j==1 and k==1)
                  //   printf("league.size=%d team.size=%d\n",
                  //          thread.league_size(),
                  //          thread.team_size());

                  y(i,j,k) = -5*x(i,j,k) +
                    (x(i-1,j,k) + x(i+1,j,k) +
                     x(i,j-1,k) + x(i,j+1,k) +
                     x(i,j,k-1) + x(i,j,k+1));
                }
              }); // end vector range
          }); // end thread range
      }); // end team policy
  } // end for irepeat
  timer.stop();

  // for (int i=13; i<17; ++i) {
  //   for (int j=13; j<17; ++j) {
  //     for (int k=13; k<17; ++k) {
  //       printf("%f ",y(i,j,k));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_range_hierarchical

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 5 bis :
 * same as version 5, but the number of teams is a parameter
 * so we must explicitly divide computation among teams.
 *
 * - a TeamPolicy        Kokkos::policy for the outer loop
 * - a TeamThreadRange   Kokkos::policy to divide middle loop over threads
 * - a ThreadVectorRange Kokkos::policy for the inner loop (for CPU vectorization or GPU warp parallelism)
 *
 * Note that this is the only test with 3d views and Kokkos::LayoutRight
 * for all devices. We try to optimize by ensuring the inner loop will
 * use contiguous memory data.
 *
 */
double test_stencil_3d_range_hierarchical2(int n, int nbTeams, int nrepeat) {

  uint64_t nbCells = n*n*n;

  // Allocate Views - enforce LayoutRight to make sure the fastest index
  // is used for inner loop iteration
  DataArrayR x("X",n,n,n);
  DataArrayR y("Y",n,n,n);

  // init 2d range policy
#if KOKKOS_VERSION_MAJOR > 3
  using Range2D = typename Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
  using Range3D = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >;
#else
  using Range2D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<2> >;
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;
#endif

  Range2D range2d( {{0,0}}, {{n,n}} );
  Range3D range3d( {{0,0,0}}, {{n,n,n}} );

  // Initialize arrays using a 3d range policy
  Kokkos::parallel_for("init", range3d,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {
                         x(i,j,k) = init_x(i,j,k);
			 y(i,j,k) = init_y(i,j,k);
		       });

  // for (int i=13; i<17; ++i) {
  //   for (int j=13; j<17; ++j) {
  //     for (int k=13; k<17; ++k) {
  //       printf("%f ",y(i,j,k));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // get prepared for TeamPolicy
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using thread_t = team_policy_t::member_type;

  // the teams league must distribute last dimension into nbTeams chuncks
  // compute chunck size per team (rounded up)
  int chunck_size_x = (n+nbTeams-1)/nbTeams;

  // total of "iterations" per team, to be divided among threads
  // of the team
  int chunk_size_per_team = chunck_size_x * n;

  // then inside a team, chunck_size is divided among threads
  // we need some index linearization here

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for(
      team_policy_t(nbTeams,
                    Kokkos::AUTO, /* team size chosen by kokkos */
                    team_policy_t::vector_length_max()),
      KOKKOS_LAMBDA(const thread_t& thread) {

        // team id
        int teamId = thread.league_rank();

        // compute i start
        int iStart = teamId*chunck_size_x;

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(thread, chunk_size_per_team),
          [=](const int &index) {

            // index goes from 0 to chunck_size_ter_team
            // re-compute i,j from index = j + n*i ; i += iStart
            int i = index / n;
            int j = index - i*n;

            i+= iStart;

            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(thread, 1, n-1),
              [=](const int &k) {

                if (i>0 and i<n-1 and
                    j>0 and j<n-1) {

                  // if (i==1 and j==1 and k==1)
                  //   printf("league.size=%d team.size=%d\n",
                  //          thread.league_size(),
                  //          thread.team_size());

                  y(i,j,k) = -5*x(i,j,k) +
                    (x(i-1,j,k) + x(i+1,j,k) +
                     x(i,j-1,k) + x(i,j+1,k) +
                     x(i,j,k-1) + x(i,j,k+1));
                }
              }); // end vector range
          }); // end thread range
      }); // end team policy
  } // end for irepeat
  timer.stop();

  // for (int i=13; i<17; ++i) {
  //   for (int j=13; j<17; ++j) {
  //     for (int k=13; k<17; ++k) {
  //       printf("%f ",y(i,j,k));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_range_hierarchical2

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 5 ter :
 * same as version 5 but use linearized array
 *
 * - a TeamPolicy        Kokkos::policy for the outer loop
 * - a TeamThreadRange   Kokkos::policy to divide middle loop over threads
 * - a ThreadVectorRange Kokkos::policy for the inner loop (for CPU vectorization or GPU warp parallelism)
 *
 * Note that we use the same linearization for both CPU/GPU :
 * Kokkos::LayoutRight
 * We try to optimize by ensuring the inner loop will
 * use contiguous memory data.
 *
 */
double test_stencil_3d_range_hierarchical3(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;

  // Allocate Views - enforce LayoutRight to make sure the fastest index
  // is used for inner loop iteration
  DataArray1d x("X",n*n*n);
  DataArray1d y("Y",n*n*n);

  // init 2d range policy
#if KOKKOS_VERSION_MAJOR > 3
  using Range2D = typename Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
  using Range3D = typename Kokkos::MDRangePolicy< Kokkos::Rank<3> >;
#else
  using Range2D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<2> >;
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;
#endif

  Range2D range2d( {{0,0}}, {{n,n}} );
  Range3D range3d( {{0,0,0}}, {{n,n,n}} );

  // Initialize arrays using a 3d range policy
  Kokkos::parallel_for("init", n*n*n,
		       KOKKOS_LAMBDA (const int& index) {
                         int i,j,k;

                         index2coord_right(index,i,j,k,n,n,n);

                         x(index) = init_x(i,j,k);
			 y(index) = init_y(i,j,k);
		       });

  // for (int i=13; i<17; ++i) {
  //   for (int j=13; j<17; ++j) {
  //     for (int k=13; k<17; ++k) {
  //       printf("%f ",y(RINDEX(i,j,k,n,n,n)));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // get prepared for TeamPolicy
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int>>;
  using thread_t = team_policy_t::member_type;
  int nbTeams = n;

  // Time computation
  Timer timer;

  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {

    // Do stencil
    Kokkos::parallel_for(
      team_policy_t(nbTeams,
                    Kokkos::AUTO, /* team size chosen by kokkos */
                    team_policy_t::vector_length_max()),
      KOKKOS_LAMBDA(const thread_t& thread) {

        int i = thread.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(thread, 1, n-1),
          [=](const int &j) {

            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(thread, 1, n-1),
              [=](const int &k) {

                if (i>0 and i<n-1 and
                    j>0 and j<n-1) {

                  y(RINDEX(i,j,k,n,n,n)) = -5*x(RINDEX(i,j,k,n,n,n)) +
                    (x(RINDEX(i-1,j,k,n,n,n)) + x(RINDEX(i+1,j,k,n,n,n)) +
                     x(RINDEX(i,j-1,k,n,n,n)) + x(RINDEX(i,j+1,k,n,n,n)) +
                     x(RINDEX(i,j,k-1,n,n,n)) + x(RINDEX(i,j,k+1,n,n,n)));
                }
              }); // end vector range
          }); // end thread range
      }); // end team policy
  } // end for irepeat
  timer.stop();

  // for (int i=13; i<17; ++i) {
  //   for (int j=13; j<17; ++j) {
  //     for (int k=13; k<17; ++k) {
  //       printf("%f ",y(RINDEX(i,j,k,n,n,n)));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }

  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;

  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  return bandwidth;

} // test_stencil_3d_range_hierarchical3

// ===============================================================
// ===============================================================
enum bench_type : int {
  SMALL,
  MEDIUM,
  LARGE
};

// ===============================================================
// ===============================================================
void bench(int nrepeat, bench_type bt) {

  constexpr int nbTests = 8;

  auto select_sizes =
    [](bench_type bt)
    {

      switch(bt) {
      case bench_type::SMALL:
        return std::vector<int> {32, 48};
      case bench_type::MEDIUM:
        return std::vector<int> {32, 48, 64, 92, 128, 160, 192, 224, 256};
      case bench_type::LARGE:
        return std::vector<int> {32, 48, 64, 92, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768};
      default:
        return std::vector<int> {32, 48};
      }
    };

  auto size_list = select_sizes(bt);
  int size = size_list.size();

  std::array<std::vector<double>,nbTests> v;

  std::vector<std::string> test_names = {
    "# test_stencil_3d_flat",
    "# test_stencil_3d_flat_1d_array",
    "# test_stencil_3d_flat_vector without views",
    "# test_stencil_3d_flat_vector with    views",
    "# test_stencil_3d_range",
    "# test_stencil_3d_range_vector",
    "# test_stencil_3d_range_hierarchical",
    "# test_stencil_3d_range_hierarchical_linearized"
  };

  std::cout << "###########################" << test_names[0] << "\n";
  for (auto n : size_list)
    v[0].push_back(test_stencil_3d_flat(n, nrepeat));

  std::cout << "###########################" << test_names[1] << "\n";
  for (auto n : size_list)
    v[1].push_back(test_stencil_3d_flat_1d_array(n, nrepeat));

  std::cout << "###########################" << test_names[2] << "\n";
  for (auto n : size_list)
    v[2].push_back(test_stencil_3d_flat_vector(n, nrepeat,false));

  std::cout << "###########################" << test_names[3] << "\n";
  for (auto n : size_list)
    v[3].push_back(test_stencil_3d_flat_vector(n, nrepeat,true));

  std::cout << "###########################" << test_names[4] << "\n";
  for (auto n : size_list)
    v[4].push_back(test_stencil_3d_range(n, nrepeat));

  std::cout << "###########################" << test_names[5] << "\n";
  for (auto n : size_list)
    v[5].push_back(test_stencil_3d_range_vector(n, nrepeat));

  std::cout << "###########################" << test_names[6] << "\n";
  for (auto n : size_list)
    v[6].push_back(test_stencil_3d_range_hierarchical(n, nrepeat));

  std::cout << "###########################" << test_names[7] << "\n";
  for (auto n : size_list)
    v[7].push_back(test_stencil_3d_range_hierarchical3(n, nrepeat));

  /*
   * create python script for plotting results
   */
  std::ofstream ofs ("plot_stencil_perf.py", std::ofstream::out);

  ofs << "import numpy as np\n";
  ofs << "import matplotlib.pyplot as plt\n";
  ofs << "from matplotlib import rc\n";
  ofs << "#rc('text', usetex=True)\n\n";

  // output size array
  ofs << "size=np.array([";
  for (int i=0; i<size; ++i)
    i<size-1 ? ofs << size_list[i] << "," : ofs << size_list[i];
  ofs << "])\n\n";

  // output bandwidth data
  for (int iv=0; iv<nbTests; ++iv) {

    ofs << test_names[iv] << "\n";
    ofs << "v" << iv << "=np.array([";
    for (int i=0; i<size; ++i)
      i<size-1 ? ofs << v[iv][i] << "," : ofs << v[iv][i];
    ofs << "])\n\n";

  }


  ofs << "plt.plot(size,v0, label='"<<test_names[0]<<"')\n";
  ofs << "plt.plot(size,v1, label='"<<test_names[1]<<"')\n";
  ofs << "plt.plot(size,v2, label='"<<test_names[2]<<"')\n";
  ofs << "plt.plot(size,v3, label='"<<test_names[3]<<"')\n";
  ofs << "plt.plot(size,v4, label='"<<test_names[4]<<"')\n";
  ofs << "plt.plot(size,v5, label='"<<test_names[5]<<"')\n";
  ofs << "plt.plot(size,v6, label='"<<test_names[6]<<"')\n";
  ofs << "plt.plot(size,v7, label='"<<test_names[7]<<"')\n";

  ofs << "plt.grid(True)\n";

  // ofs << "plt.title('3d Heat kernel performance on Skylake (1 socket, irene)')\n"
  ofs << "plt.title('3d 7 points stencil kernel performance')\n";
  ofs << "plt.xlabel('N - linear size')\n";
  ofs << "plt.ylabel(r'Bandwidth (GBytes/s)')\n";

  ofs << "plt.legend()\n";
  ofs << "plt.show()\n";

  ofs.close();

} // bench

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char* argv[]) {

  // Parameters
  int n = 256;       // 3d array linear size
  int nrepeat = 10;  // number of kernel invocations
  int nteams = 4;   // default number of teams (for TeamPolicy)
  bool bench_enabled = false; // run bench instead
  bench_type bt = bench_type::MEDIUM;

  // Read command line arguments
  for(int i=0; i<argc; i++) {
    if( strcmp(argv[i], "-n") == 0) {
      n = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nteams") == 0) {
      nteams = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-b") == 0) {
      bench_enabled = true;
    } else if( strcmp(argv[i], "-bs") == 0) {
      bench_enabled = true;
      bt = bench_type::SMALL;
    } else if( strcmp(argv[i], "-bm") == 0) {
      bench_enabled = true;
      bt = bench_type::MEDIUM;
    } else if( strcmp(argv[i], "-bl") == 0) {
      bench_enabled = true;
      bt = bench_type::LARGE;
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("STENCIL 3D Options:\n");
      printf("  -n <int>:         3d linear size (default: 256)\n");
      printf("  -nrepeat <int>:   number of integration invocations (default: 10)\n");
      printf("  -nteams <int>:    number of teams (only for version 5bis, default is 4)\n");
      printf("  -b :              run full benchmark (default MEDIUM size)\n");
      printf("  -bs :             run full benchmark (SMALL size)\n");
      printf("  -bm :             run full benchmark (MEDIUM size)\n");
      printf("  -bl :             run full benchmark (LARGE size)\n");
      printf("  -help (-h):       print this message\n");
      return EXIT_SUCCESS;
    }
  }

  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  std::cout << "##########################\n";
  std::cout << "KOKKOS CONFIG             \n";
  std::cout << "##########################\n";

  std::ostringstream msg;
  std::cout << "Kokkos configuration" << std::endl;
  if ( Kokkos::hwloc::available() ) {
    msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
	<< "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
	<< "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
	<< "] )"
	<< std::endl ;
  }
  Kokkos::print_configuration( msg );
  std::cout << msg.str();
  std::cout << "##########################\n";

  if (bench_enabled) {

    bench(nrepeat,bt);

  } else {

    // run test
    std::cout << "========================================\n";
    std::cout << "reference naive test using 1d flat range\n";
    test_stencil_3d_flat(n, nrepeat);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 1d flat range 1d array\n";
    test_stencil_3d_flat_1d_array(n, nrepeat);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 2d flat range and vectorization (no views)\n";
    test_stencil_3d_flat_vector(n, nrepeat,false);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 2d flat range and vectorization (with views)\n";
    test_stencil_3d_flat_vector(n, nrepeat,true);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 3d range\n";
    test_stencil_3d_range(n, nrepeat);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 3d range and vectorization\n";
    test_stencil_3d_range_vector(n, nrepeat);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 3d range and vectorization with team policy\n";
    test_stencil_3d_range_hierarchical(n, nrepeat);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 3d range and vectorization with team policy - nbTeams configurable\n";
    test_stencil_3d_range_hierarchical2(n, nteams, nrepeat);

    std::cout << "========================================\n";
    std::cout << "reference naive test using 3d range and vectorization with team policy - linearized data\n";
    test_stencil_3d_range_hierarchical3(n, nrepeat);
  }

  // Shutdown Kokkos
  Kokkos::finalize();

  return EXIT_SUCCESS;
}
