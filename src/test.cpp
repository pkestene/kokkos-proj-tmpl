#include <Kokkos_Core.hpp>

#include <iostream>

KOKKOS_FUNCTION
constexpr Kokkos::Array<double, 5000>
compute_coefs()
{
  Kokkos::Array<long double, 5> x{ 1., 2., 3., 4., 5. };
  Kokkos::Array<double, 5000>   y{};
  for (int i = 0; i < y.size(); ++i)
    y[i] = static_cast<double>(x[i % 5]);
  return y;
}

int
main()
{
  Kokkos::initialize();

  {
    Kokkos::View<double *> result{ "", 50 };

    constexpr auto magic_coefs = compute_coefs();
    Kokkos::parallel_for(
      "Dummy", result.size(), KOKKOS_LAMBDA(int i) { result(i) = magic_coefs[i]; });

    auto host_copy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);

    for (int i = 0; i < result.size(); ++i)
      std::cout << host_copy(i) << '\n';
  }

  Kokkos::finalize();

  return 0;
}
