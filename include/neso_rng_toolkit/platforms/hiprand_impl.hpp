#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_HIPRAND_IMPL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_HIPRAND_IMPL_HPP_
#ifdef NESO_RNG_TOOLKIT_HIPRAND

#include "hiprand.hpp"
#include <hiprand/hiprand.hpp>
#include <type_traits>

namespace NESO::RNGToolkit {

#ifdef __HIP_PLATFORM_AMD__
#endif
#ifdef __HIP_PLATFORM_NVIDIA__
#endif

template <typename VALUE_TYPE>
RNGSharedPtr<VALUE_TYPE> hipRANDPlatform<VALUE_TYPE>::create_rng(
    [[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
    std::uint64_t seed, sycl::device device, std::size_t device_index,
    std::string generator_name) {
  generator_name = this->get_generator_name(generator_name, "default");
  if (this->check_generator_name(generator_name, this->generators)) {
    return nullptr;
  } else {
    return nullptr;
  }
}

template <typename VALUE_TYPE>
RNGSharedPtr<VALUE_TYPE> hipRANDPlatform<VALUE_TYPE>::create_rng(
    [[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
    std::uint64_t seed, [[maybe_unused]] sycl::device device,
    std::size_t device_index, std::string generator_name) {
  generator_name = this->get_generator_name(generator_name, "default");
  if (this->check_generator_name(generator_name, this->generators)) {
    return nullptr;
  } else {
    return nullptr;
  }
}

} // namespace NESO::RNGToolkit
#endif
#endif
