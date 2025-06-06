#include <gtest/gtest.h>
#include <neso_rng_toolkit.hpp>

using namespace NESO::RNGToolkit;

TEST(RNGToolkit, print_version) {
  print_version();

#ifdef NESO_RNG_TOOLKIT_ONEDPL
  std::cout << "oneDPL enabled" << std::endl;
#endif

  std::cout << "default platform: " << get_default_platform() << std::endl;
}
