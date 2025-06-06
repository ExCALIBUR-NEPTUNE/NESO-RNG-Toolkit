#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_ONEDPL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_ONEDPL_HPP_

#include "../platform.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"

#ifdef NESO_RNG_TOOLKIT_ONEDPL
#include <oneapi/dpl/random>
#endif

namespace NESO::RNGToolkit {

#ifdef NESO_RNG_TOOLKIT_ONEDPL
/**
 * This is the main interface to the oneDPL random implementations.
 */
template <typename VALUE_TYPE>
struct OneDPLPlatform : public Platform<VALUE_TYPE> {
  virtual ~OneDPLPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index) override {
    sycl::queue queue(device);
    return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
        std::make_shared<StdLibRNG<VALUE_TYPE, std::mt19937_64,
                                   std::uniform_real_distribution<VALUE_TYPE>>>(
            queue, seed, std::uniform_real_distribution<VALUE_TYPE>(0.0, 1.0)));
  }

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index) override {
    sycl::queue queue(device);
    return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
        std::make_shared<StdLibRNG<VALUE_TYPE, std::mt19937_64,
                                   std::normal_distribution<VALUE_TYPE>>>(
            queue, seed, std::normal_distribution<VALUE_TYPE>(0.0, 1.0)));
  }
};
#else
/**
 * If oneDPL is not found then we make the OneDPLPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using OneDPLPlatform = StdLibPlatform<VALUE_TYPE>;
#endif

} // namespace NESO::RNGToolkit

#endif
