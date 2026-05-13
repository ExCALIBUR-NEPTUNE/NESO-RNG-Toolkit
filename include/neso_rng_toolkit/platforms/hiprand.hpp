#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_HIPRAND_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_HIPRAND_HPP_

#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"

namespace NESO::RNGToolkit {
#ifdef NESO_RNG_TOOLKIT_HIPRAND

/**
 * Try to determine if the SYCL device is actually a HIP device.
 *
 * @param device SYCL device.
 * @param device_index Device index in host sycl platform.
 * @returns True if this function determines that the device is a HIP device.
 */
bool is_hip_device(sycl::device device, const std::size_t device_index);

/**
 * This is the main interface to the hipRAND random implementations.
 */
template <typename VALUE_TYPE>
struct hipRANDPlatform : public Platform<VALUE_TYPE> {

  const static inline std::set<std::string> generators = {"default_engine"};

  virtual ~hipRANDPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index,
             std::string generator_name) override;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index,
             std::string generator_name) override;
};

extern template struct hipRANDPlatform<double>;
extern template struct hipRANDPlatform<float>;

#else

/**
 * If hipRAND is not found then we make the hipRANDPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using hipRANDPlatform = StdLibPlatform<VALUE_TYPE>;
#endif

} // namespace NESO::RNGToolkit

#endif
