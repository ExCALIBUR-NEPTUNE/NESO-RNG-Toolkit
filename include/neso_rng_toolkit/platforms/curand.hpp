#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_HPP_

#include "../distribution.hpp"
#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"
#include <functional>
#include <iostream>
#include <map>
#include <sycl/sycl.hpp>

namespace NESO::RNGToolkit {
#ifdef NESO_RNG_TOOLKIT_CURAND

/**
 * Try to determine if the SYCL device is actually a cuda device.
 *
 * @param device SYCL device.
 * @param device_index Device index in host sycl platform.
 * @returns True if this function determines that the device is a CUDA device.
 */
bool is_cuda_device(sycl::device device, const std::size_t device_index);

/**
 * This is the main interface to the curand random implementations.
 */
template <typename VALUE_TYPE>
struct CurandPlatform : public Platform<VALUE_TYPE> {
  static const inline std::set<std::string> generators = {"default"};

  virtual ~CurandPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device, std::size_t device_index,
             std::string generator_name) override;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
             std::uint64_t seed, [[maybe_unused]] sycl::device device,
             std::size_t device_index, std::string generator_name) override;
};

extern template struct CurandPlatform<double>;
extern template struct CurandPlatform<float>;

#else

inline bool is_cuda_device(sycl::device, const std::size_t) { return false; }

/**
 * If curand is not found then we make the CurandPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using CurandPlatform = StdLibPlatform<VALUE_TYPE>;

#endif
} // namespace NESO::RNGToolkit

#endif
