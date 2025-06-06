#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_STDLIB_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_STDLIB_HPP_

#include "../platform.hpp"

namespace NESO::RNGToolkit {

/**
 * This is the main interface to the C++ stdlib random implementations.
 */
template <typename VALUE_TYPE>
struct StdLibPlatform : public Platform<VALUE_TYPE> {

  /*
   * Create an RNG instance.
   *
   * @param distribution Distribution RNG samples should be from.
   * @param seed Value to seed RNG with.
   * @param device SYCL Device samples are to be created on.
   * @param device_index Index of SYCL device on the SYCL platform.
   * @returns RNG instance. nullptr on Error.
   */
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Uniform distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index) override {}

  /*
   * Create an RNG instance.
   *
   * @param distribution Distribution RNG samples should be from.
   * @param seed Value to seed RNG with.
   * @param device SYCL Device samples are to be created on.
   * @param device_index Index of SYCL device on the SYCL platform.
   * @returns RNG instance. nullptr on Error.
   */
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Normal distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index) override {}
};

} // namespace NESO::RNGToolkit

#endif
