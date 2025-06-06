#include <gtest/gtest.h>
#include <neso_rng_toolkit.hpp>
#include <set>
#include <vector>

using namespace NESO::RNGToolkit;

TEST(RNGToolkit, print_version) {
  print_version();

#ifdef NESO_RNG_TOOLKIT_ONEDPL
  std::cout << "oneDPL enabled" << std::endl;
#endif

  std::cout << "default platform: " << get_default_platform() << std::endl;
}

TEST(RNGToolkit, create_seeds) {

  std::size_t N = 127;
  std::uint64_t base_seed = 1284124;

  std::vector<std::uint64_t> seeds(N);
  std::set<std::uint64_t> seeds_set;

  for (std::size_t ix = 0; ix < N; ix++) {
    seeds[ix] = create_seeds(N, ix, base_seed);
    seeds_set.insert(seeds[ix]);
  }

  ASSERT_EQ(seeds_set.size(), N);

  for (std::size_t jx = 0; jx < N; jx++) {
    for (std::size_t ix = 0; ix < jx + 1; ix++) {
      auto seedx = create_seeds(N, ix, base_seed);
      ASSERT_EQ(seedx, seeds[ix]);
    }
  }
}
