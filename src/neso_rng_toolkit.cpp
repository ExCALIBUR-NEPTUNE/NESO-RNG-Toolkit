#include <iostream>
#include <neso_rng_toolkit.hpp>

namespace NESO::RNGToolkit {

void print_version() {
  std::cout << NESO_RNG_TOOLKIT_VERSION_MAJOR << "."
            << NESO_RNG_TOOLKIT_VERSION_MINOR << "."
            << NESO_RNG_TOOLKIT_VERSION_PATCH << std::endl;

#ifdef NESO_RNG_TOOLKIT_ONEMKL
  std::cout << "oneMKL enabled" << std::endl;
#endif

#ifdef NESO_RNG_TOOLKIT_CURAND
  std::cout << "curand enabled" << std::endl;
#endif
}

} // namespace NESO::RNGToolkit
