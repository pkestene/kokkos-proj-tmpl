#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>

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
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "OpenMPTimer.h"
using Timer = OpenMPTimer;
#else
#include "SimpleTimer.h"
using Timer = SimpleTimer;
#endif

using Device = Kokkos::DefaultExecutionSpace;
using DataArray = Kokkos::View<real_t***, Device>;

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
}

// ===============================================================
// ===============================================================
KOKKOS_INLINE_FUNCTION
int coord2index(int i,  int j,  int k,
                int Nx, int Ny, int Nz)
{
  UNUSED(Nx);
  UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
  return i + Nx*j + Nx*Ny*k; // left layout
#else
  return k + Nz*j + Nz*Ny*i; // right layout
#endif
}

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 1 : naive
 * - data is a 3d array
 * - all loops parallelized with a single parallel_for
 */
void test_stencil_3d_flat(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;
  
  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);
  
  // Initialize arrays
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA (const int& index) {
      int i,j,k;
      
      index2coord(index,i,j,k,n,n,n);
      
      x(i,j,k) = 1.0*(i+j+k+0.1);
      y(i,j,k) = 3.0*(i+j+k);
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
  
  // Print results
  double time_seconds = timer.elapsed();

  // 6+1 reads + 1 write
  double dataSizeMBytes = 1.0e-6*nbCells*sizeof(real_t)*(7+1);
  double bandwidth = 1.0e-3*dataSizeMBytes*nrepeat/time_seconds;
  
  printf("#nbCells      Time(s)  TimePerIterations(s) size(MB) BW(GB/s)\n");
  printf("%13lu %8lf %20.3e  %6.3f %3.3f\n",
	 nbCells, time_seconds, time_seconds/nrepeat,
	 dataSizeMBytes,bandwidth);

  
} // test_stencil_3d_flat

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 2:
 * - data is a 3d array
 * - only loops over i,j are parallelized, loop over k is kept inside
 * - optionally, on can use 1D subview to access data, and help the compiler
 *   to recognize a vectorizable loop
 *
 *
 * subview's are important here. 
 * Without 1d views, you'll have about 25% percent perfomance drop here.
 *
 * Use parameter use_1d_views to activate/deactivate the use of 1d views.
 */
void test_stencil_3d_flat_vector(int n, int nrepeat, bool use_1d_views) {

  uint64_t nbCells = n*n*n;

  uint64_t nbIter = n*n;
  
  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);
  
  // Initialize arrays
  Kokkos::parallel_for(nbCells, KOKKOS_LAMBDA (const int& index) {
      int i,j,k;
      
      index2coord(index,i,j,k,n,n,n);
      
      x(i,j,k) = 1.0*(i+j+k+0.1);
      y(i,j,k) = 3.0*(i+j+k);
    });

  // Time computation
  Timer timer;
  
  timer.start();

  if (use_1d_views) {

    for(int irepeat = 0; irepeat < nrepeat; irepeat++) {
      
      // Do stencil
      Kokkos::parallel_for(nbIter, KOKKOS_LAMBDA (const int& index) {
	  int i,j;
	  //index2coord(index,i,j,k,n,n,n);
	  
	  // index = j + n * i
	  i = index / n;
	  j = index - i*n;
	  
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
	  
	  // index = j + n * i
	  i = index / n;
	  j = index - i*n;	  
	  
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

  
} // test_stencil_3d_flat_vector

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 3 :
 * same as version 1 (naive) but uses a 3d range policy.
 */
void test_stencil_3d_range(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;
  
  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // init 3d range policy
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;

  Range3D range( {{0,0,0}}, {{n,n,n}} );

  
  // Initialize arrays
  Kokkos::parallel_for("init", range,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {      
			 x(i,j,k) = 1.0*(i+j+k+0.1);
			 y(i,j,k) = 3.0*(i+j+k);
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

  
} // test_stencil_3d_range

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 4 :
 * same as version 3 but uses a 2d Range policy and keep the loop over
 * index k inside kernel.
 */
void test_stencil_3d_range_vector(int n, int nrepeat) {

  uint64_t nbCells = n*n*n;
  
  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // init 2d range policy
  using Range2D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<2> >;
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;

  Range2D range2d( {{0,0}}, {{n,n}} );
  Range3D range3d( {{0,0,0}}, {{n,n,n}} );

  
  // Initialize arrays
  Kokkos::parallel_for("init", range3d,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {      
			 x(i,j,k) = 1.0*(i+j+k+0.1);
			 y(i,j,k) = 3.0*(i+j+k);
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

  
} // test_stencil_3d_range_vector

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * version 5 :
 * same as version 4 but uses Hierarchical parallelism, i.e.
 * - a TeamPolicy        Kokkos::policy for the outer loop 
 * - a ThreadVectorRange Kokkos::policy for the inner loop (to be vectorized)
 *
 */
void test_stencil_3d_range_vector2(int n, int nrepeat, int nbTeams) {

  uint64_t nbCells = n*n*n;
  
  // Allocate Views
  DataArray x("X",n,n,n);
  DataArray y("Y",n,n,n);

  // init 2d range policy
  using Range2D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<2> >;
  using Range3D = typename Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank<3> >;

  Range2D range2d( {{0,0}}, {{n,n}} );
  Range3D range3d( {{0,0,0}}, {{n,n,n}} );

  
  // Initialize arrays using a 3d range policy
  Kokkos::parallel_for("init", range3d,
		       KOKKOS_LAMBDA (const int& i,
				      const int& j,
				      const int& k) {      
			 x(i,j,k) = 1.0*(i+j+k+0.1);
			 y(i,j,k) = 3.0*(i+j+k);
		       });

  // get prepared for TeamPolicy
  using team_member_t = typename Kokkos::TeamPolicy<>::member_type;

  
  // Time computation
  Timer timer;
  
  timer.start();
  for(int irepeat = 0; irepeat < nrepeat; irepeat++) {
    
    // Do stencil
    Kokkos::parallel_for
      ("stencil compute - 3d range vector",
       Kokkos::TeamPolicy<> (nbTeams, Kokkos::AUTO),
       KOKKOS_LAMBDA (team_member_t team_member) {

	int i = team_member.team_rank();
	int j = team_member.league_rank();

	int ni = (n+team_member.team_size()  -1)/team_member.team_size();
	int nj = (n+team_member.league_size()-1)/team_member.league_size();
	
	// make sure indexes (i,j) cover the full (nx,ny) range
	for (int jj=0, j=team_member.league_rank();
	     jj<nj;
	     ++jj, j+=team_member.league_size()) {
	  for (int ii=0, i=team_member.team_rank();
	       ii<ni;
	       ++ii, i+=team_member.team_size()) {

	    auto x_i_j = Kokkos::subview(x, i, j, Kokkos::ALL());
	    auto y_i_j = Kokkos::subview(y, i, j, Kokkos::ALL());
	    
	    auto x_im1_j = Kokkos::subview(x, i-1, j, Kokkos::ALL());
	    auto x_ip1_j = Kokkos::subview(x, i+1, j, Kokkos::ALL());
	    
	    auto x_i_jm1 = Kokkos::subview(x, i, j-1, Kokkos::ALL());
	    auto x_i_jp1 = Kokkos::subview(x, i, j+1, Kokkos::ALL());
	    
	    if (i>0 and i<n-1 and
		j>0 and j<n-1) {

	      // Kokkos::parallel_for
	      // 	(Kokkos::ThreadVectorRange(team_member, 1, n-1),
	      // 	 KOKKOS_LAMBDA(int& k) {

	      // 	  y_i_j(k) = -5*x_i_j(k) +
	      // 	  ( x_im1_j(k) + x_ip1_j(k) +
	      // 	    x_i_jm1(k) + x_i_jp1(k) +
	      // 	    x_i_j(k-1) + x_i_j(k+1) );
		  
	      // 	});
	      
	      // vectorization loop
#if defined( __INTEL_COMPILER )
#pragma ivdep
#endif
	      for (int k=1; k<n-1; ++k)
	      	y_i_j(k) = -5*x_i_j(k) +
	      	  ( x_im1_j(k) + x_ip1_j(k) +
	      	    x_i_jm1(k) + x_i_jp1(k) +
	      	    x_i_j(k-1) + x_i_j(k+1) );
		// y(i,j,k) = -5*x(i,j,k) +
		//   ( x(i-1,j,k) + x(i+1,j,k) +
		//     x(i,j-1,k) + x(i,j+1,k) +
		//     x(i,j,k-1) + x(i,j,k+1) );
	      
	    }
	  } // end for ii
	} // end for jj
	
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

  
} // test_stencil_3d_range_vector2

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char* argv[]) {

  // Parameters
  int n = 256;       // 3d array linear size 
  int nrepeat = 10;  // number of kernel invocations
  int nteams = 64;   // default number of teams (for TeamPolicy)
  
  // Read command line arguments
  for(int i=0; i<argc; i++) {
    if( strcmp(argv[i], "-n") == 0) {
      n = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nrepeat") == 0) {
      nrepeat = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nteams") == 0) {
      nteams = atoi(argv[++i]);      
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("STENCIL 3D Options:\n");
      printf("  -n <int>:         3d linear size (default: 256)\n");
      printf("  -nrepeat <int>:   number of integration invocations (default: 10)\n");
      printf("  -help (-h):       print this message\n");
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
  
  // run test
  std::cout << "========================================\n";
  std::cout << "reference naive test using 1d flat range\n";
  test_stencil_3d_flat(n, nrepeat);

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
  test_stencil_3d_range_vector2(n, nrepeat,nteams);

  // Shutdown Kokkos
  Kokkos::finalize();
}
