#ifndef _NESO_RNG_TOOLKIT_CREATE_RNG_HPP_
#define _NESO_RNG_TOOLKIT_CREATE_RNG_HPP_

#include "platforms/curand.hpp"
#include "platforms/onemkl.hpp"
#include "platforms/stdlib.hpp"
#include "rng.hpp"

namespace NESO::RNGToolkit {

/**
 * @returns The default platform name.
 */
std::string get_default_platform();

/**
 * Create N seeds, e.g. for N MPI ranks and returns the i-th.
 *
 * @param size Number of seeds to create, e.g. number of MPI ranks.
 * @param rank Index to return from this call, e.g. the MPI rank.
 * @param seed Base seed to generate seeds from, this should be identical to all
 * calls of this function.
 * @returns The seed indicated by the rank argument.
 */
std::uint64_t create_seeds(std::size_t size, std::size_t rank,
                           std::uint64_t seed);

/**
 * This is the function users could call to create a RNG instance.
 *
 * @param distribution Distribution RNG samples should be from.
 * @param seed Value to seed RNG with.
 * @param device SYCL Device samples are to be created on.
 * @param device_index Index of SYCL device on the SYCL platform.
 * @param platform_name Name of preferred RNG platform, default="default".
 * @param generator_name Name of preferred RNG generator method,
 * default="default".
 * @returns RNG instance. nullptr on Error.
 */
template <typename VALUE_TYPE, typename DISTRIBUTION_TYPE>
[[nodiscard]] RNGSharedPtr<VALUE_TYPE>
create_rng(DISTRIBUTION_TYPE distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name = "default",
           std::string generator_name = "default") {
  RNGSharedPtr<VALUE_TYPE> rng = nullptr;

  if (platform_name == "default") {
    platform_name = get_default_platform();
  }
  platform_name =
      Private::get_env_string("NESO_RNG_TOOLKIT_PLATFORM", platform_name);

  generator_name =
      Private::get_env_string("NESO_RNG_TOOLKIT_GENERATOR", generator_name);

  if (platform_name == "stdlib") {
    rng = StdLibPlatform<VALUE_TYPE>{}.create_rng(distribution, seed, device,
                                                  device_index, generator_name);
  }

  if (platform_name == "oneMKL") {
    rng = OneMKLPlatform<VALUE_TYPE>{}.create_rng(distribution, seed, device,
                                                  device_index, generator_name);
  }

  if (platform_name == "curand") {
    if (is_cuda_device(device, device_index)) {
      rng = CurandPlatform<VALUE_TYPE>{}.create_rng(
          distribution, seed, device, device_index, generator_name);
    } else {
      rng = StdLibPlatform<VALUE_TYPE>{}.create_rng(
          distribution, seed, device, device_index, generator_name);
    }
  }

  if (rng == nullptr) {
    std::cout << "Unknown RNG platform: " << platform_name << std::endl;
  } else if (Private::get_env_size_t("NESO_RNG_TOOLKIT_PLATFORM_VERBOSE", 0)) {
    std::cout << "NESO-RNG-Toolkit RNG Platform: " << rng->platform_name
              << std::endl;
  }

  return rng;
}

extern template RNGSharedPtr<double>
create_rng(Distribution::Uniform<double> distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name, std::string generator_name);

extern template RNGSharedPtr<double>
create_rng(Distribution::Normal<double> distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name, std::string generator_name);

} // namespace NESO::RNGToolkit

#endif
