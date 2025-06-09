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

TEST(RNGToolkit, next_value) {

  constexpr double smallest_normal_pos = 2.2250738585072014e-308;
  constexpr double smallest_normal_neg = -2.2250738585072014e-308;

  ASSERT_TRUE(std::isnormal(smallest_normal_pos));
  ASSERT_TRUE(std::isnormal(smallest_normal_neg));

  for (double vx : {-2000.0, -1.0, -1.1, smallest_normal_neg, 0.0,
                    smallest_normal_pos, 1.0, 2.1, 2000.0}) {
    double v = Distribution::next_value(vx);
    ASSERT_TRUE(std::isnormal(v) || v == 0.0);
    ASSERT_TRUE(v > vx);

    const double vv = std::nextafter(vx, std::numeric_limits<double>::max());
    if (std::isnormal(vv)) {
      ASSERT_EQ(v, vv);
    }
  }
}

TEST(RNGToolkit, previous_value) {

  constexpr double smallest_normal_pos = 2.2250738585072014e-308;
  constexpr double smallest_normal_neg = -2.2250738585072014e-308;

  ASSERT_TRUE(std::isnormal(smallest_normal_pos));
  ASSERT_TRUE(std::isnormal(smallest_normal_neg));

  for (double vx : {-2000.0, -1.0, -1.1, smallest_normal_neg, 0.0,
                    smallest_normal_pos, 1.0, 2.1, 2000.0}) {
    double v = Distribution::previous_value(vx);
    ASSERT_TRUE(std::isnormal(v) || v == 0.0);
    ASSERT_TRUE(v < vx);

    const double vv = std::nextafter(vx, std::numeric_limits<double>::lowest());
    if (std::isnormal(vv)) {
      ASSERT_EQ(v, vv);
    }
  }
}
