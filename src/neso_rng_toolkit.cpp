#include <iostream>
#include <neso_rng_toolkit.hpp>

namespace NESO::RNGToolkit {

void print_version() {
  std::cout << NESO_RNG_TOOLKIT_VERSION_MAJOR << "."
            << NESO_RNG_TOOLKIT_VERSION_MINOR << "."
            << NESO_RNG_TOOLKIT_VERSION_PATCH << std::endl;
}

} // namespace NESO::RNGToolkit
