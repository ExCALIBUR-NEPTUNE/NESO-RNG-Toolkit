#ifndef _NESO_RNG_TOOLKIT_PLATFORM_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORM_HPP_

#include "distribution.hpp"
#include "rng.hpp"
#include "typedefs.hpp"

namespace NESO::RNGToolkit {

/**
 * This is the main interface to abstract the different platforms/vendors.
 */
template <typename VALUE_TYPE> struct Platform {
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Uniform distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index) = 0;

  virtual RNGSharedPtr<VALUE_TYPE> create_rng(Distribution::Normal distribution,
                                              std::uint64_t seed,
                                              sycl::device device,
                                              std::size_t device_index) = 0;
};

} // namespace NESO::RNGToolkit

#endif
