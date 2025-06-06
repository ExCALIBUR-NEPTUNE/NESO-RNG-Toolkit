#ifndef _NESO_RNG_TOOLKIT_CREATE_RNG_HPP_
#define _NESO_RNG_TOOLKIT_CREATE_RNG_HPP_

#include "platforms/stdlib.hpp"
#include "rng.hpp"

namespace NESO::RNGToolkit {

/**
 * @returns The default platform name.
 */
std::string get_default_platform();

/**
 * This is the function users could call to create and RNG instance.
 *
 * @param distribution Distribution RNG samples should be from.
 * @param seed Value to seed RNG with.
 * @param device SYCL Device samples are to be created on.
 * @param device_index Index of SYCL device on the SYCL platform.
 * @param platform_name Name of preferred RNG platform, default="default".
 * @returns RNG instance. nullptr on Error.
 */
template <typename VALUE_TYPE, typename DISTRIBUTION_TYPE>
[[nodiscard]] RNGSharedPtr<VALUE_TYPE>
create_rng(DISTRIBUTION_TYPE distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name = "default") {
  RNGSharedPtr<VALUE_TYPE> rng = nullptr;

  if (platform_name == "default") {
    platform_name = get_default_platform();
  }
  platform_name = get_env_string("NESO_RNG_TOOLKIT_PLATFORM", platform_name);

  if (platform_name == "stdlib") {
    return StdLibPlatform<VALUE_TYPE>{}.create_rng(distribution, seed, device,
                                                   device_index);
  }

  return nullptr;
}

} // namespace NESO::RNGToolkit

#endif
