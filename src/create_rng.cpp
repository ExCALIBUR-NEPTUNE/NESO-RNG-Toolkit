#include <neso_rng_toolkit/create_rng.hpp>

namespace NESO::RNGToolkit {

/**
 * @returns The default platform name.
 */
std::string get_default_platform() {
#ifdef NESO_RNG_TOOLKIT_ONEDPL
  return "onedpl";
#else
  return "stdlib";
#endif
}

} // namespace NESO::RNGToolkit
