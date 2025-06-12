#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_ONEMKL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_ONEMKL_HPP_

#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"

namespace NESO::RNGToolkit {

#ifdef NESO_RNG_TOOLKIT_ONEMKL

/**
 * This is the main interface to the oneMKL random implementations.
 */
template <typename VALUE_TYPE>
struct OneMKLPlatform : public Platform<VALUE_TYPE> {

  const static inline std::set<std::string> generators = {"default_engine"};

  virtual ~OneMKLPlatform() = default;

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

extern template struct OneMKLPlatform<double>;
extern template struct OneMKLPlatform<float>;

#else

/**
 * If oneMKL is not found then we make the OneMKLPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using OneMKLPlatform = StdLibPlatform<VALUE_TYPE>;
#endif

} // namespace NESO::RNGToolkit

#endif
