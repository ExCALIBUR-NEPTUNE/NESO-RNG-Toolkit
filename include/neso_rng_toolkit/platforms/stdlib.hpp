#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_STDLIB_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_STDLIB_HPP_

#include "../platform.hpp"

namespace NESO::RNGToolkit {

/**
 * This is the main interface to abstract the different platforms/vendors.
 */
template <typename VALUE_TYPE> struct StdLibPlatform : public Platform<VALUE_TYPE> {
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Uniform distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index) override {



  }

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Normal distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index) override {


  }
};


} // namespace NESO::RNGToolkit

#endif
