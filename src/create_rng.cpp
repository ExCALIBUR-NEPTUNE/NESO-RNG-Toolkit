#include <neso_rng_toolkit/create_rng.hpp>
#include <random>
#include <set>

namespace NESO::RNGToolkit {

std::string get_default_platform() {
#ifdef NESO_RNG_TOOLKIT_ONEMKL
  return "oneMKL";
#else

#ifdef NESO_RNG_TOOLKIT_CURAND
  return "curand";
#else
  return "stdlib";
#endif

#endif
}

std::uint64_t create_seeds(std::size_t size, std::size_t rank,
                           std::uint64_t seed) {
  auto rng = std::mt19937_64(seed);
  auto dist = std::uniform_int_distribution<std::uint64_t>(
      0, std::numeric_limits<std::uint64_t>::max());

  std::set<std::uint64_t> values_set;
  std::uint64_t seed_out = 0;
  for (std::size_t rx = 0; rx < (rank + 1); rx++) {
    do {
      seed_out = dist(rng);
    } while (values_set.count(seed_out));
    values_set.insert(seed_out);
  }

  return seed_out;
}

template RNGSharedPtr<double>
create_rng(Distribution::Uniform<double> distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name, std::string generator_name);

template RNGSharedPtr<double>
create_rng(Distribution::Normal<double> distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name, std::string generator_name);

} // namespace NESO::RNGToolkit
